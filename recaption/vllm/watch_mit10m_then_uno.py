#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent.parent
LOG_DIR = WORKDIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = LOG_DIR / "mit10m_watch_state.json"
LOG_PATH = LOG_DIR / "mit10m_watch.log"
RUN_SCRIPT = WORKDIR / "run.slurm"
REPAIR_SCRIPT = WORKDIR / "run_explicit_tasks.slurm"
UNO_SCRIPT = WORKDIR / "scripts" / "prepare_uno_1m_stage2_manifest.py"

LOADER_MODULE = "loader_mit_10m_qwen_from_text"
WORKERS_PER_NODE = 4
LOGICAL_TASK_COUNT = 64
EXPECTED_FINAL_FILES = 64
EXPECTED_LINES = 840855
POLL_SECONDS = 180
HANG_MINUTES = 20
SUSPECT_POLLS_REQUIRED = 2
CPUS_PER_WORKER = 72

WAVES = [
    ["en", "de", "zh", "ja"],
    ["fr", "es", "it", "pt"],
    ["ko", "ru", "ar", "tr"],
    ["th", "hi"],
]
ALL_LANGS = [lang for wave in WAVES for lang in wave]
LANG_TO_WAVE = {lang: wave_index for wave_index, wave in enumerate(WAVES, start=1) for lang in wave}

FULL_JOB_RE = re.compile(r"^recaption-mit-10m-([a-z]+)$")
REPAIR_JOB_RE = re.compile(r"^recaption-mit-10m-([a-z]+)-repair$")


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_cmd(args: list[str], check: bool = True) -> str:
    result = subprocess.run(
        args,
        cwd=WORKDIR,
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "suspected_slots": {},
        "wave_requires_manual_release": {},
        "latest_full_jobs": {},
        "latest_repair_jobs": {},
        "uno_started": False,
    }


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def parse_squeue() -> dict:
    output = run_cmd(
        [
            "squeue",
            "-h",
            "-u",
            os.environ.get("USER", "user"),
            "-o",
            "%i|%j|%t|%M|%R",
        ]
    )
    entries = []
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        job_id, name, state, elapsed, reason = raw_line.split("|", 4)
        entries.append(
            {
                "job_id": job_id,
                "name": name,
                "state": state,
                "elapsed": elapsed,
                "reason": reason,
            }
        )
    return {
        "entries": entries,
        "by_name": defaultdict(list),
        "running_slots": defaultdict(set),
        "base_states": {},
    }


def enrich_squeue(parsed: dict) -> dict:
    for entry in parsed["entries"]:
        parsed["by_name"][entry["name"]].append(entry)
        base_job_id = entry["job_id"].split("_", 1)[0]
        parsed["base_states"].setdefault(base_job_id, set()).add(entry["state"])
        if "_" in entry["job_id"] and not entry["job_id"].endswith("]"):
            slot = entry["job_id"].split("_", 1)[1]
            if slot.isdigit():
                parsed["running_slots"][base_job_id].add(int(slot))
    return parsed


def parse_sacct_states(job_id: str) -> list[str]:
    output = run_cmd(
        [
            "sacct",
            "-X",
            "-j",
            job_id,
            "--format=JobIDRaw,State",
            "-P",
            "-n",
        ],
        check=False,
    )
    states = []
    for line in output.splitlines():
        if not line.strip():
            continue
        _, state = line.split("|", 1)
        states.append(state)
    return states


def is_failure_state(state: str) -> bool:
    return state.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"))


def discover_jobs(parsed: dict, state: dict) -> None:
    latest_full_jobs = state.setdefault("latest_full_jobs", {})
    latest_repair_jobs = state.setdefault("latest_repair_jobs", {})

    for entry in parsed["entries"]:
        full_match = FULL_JOB_RE.match(entry["name"])
        if full_match:
            lang = full_match.group(1)
            if lang in ALL_LANGS:
                latest_full_jobs[lang] = max(
                    int(entry["job_id"].split("_", 1)[0]),
                    int(latest_full_jobs.get(lang, 0) or 0),
                )
            continue

        repair_match = REPAIR_JOB_RE.match(entry["name"])
        if repair_match:
            lang = repair_match.group(1)
            latest_repair_jobs[lang] = max(
                int(entry["job_id"].split("_", 1)[0]),
                int(latest_repair_jobs.get(lang, 0) or 0),
            )


def final_output_path(lang: str, task_id: int) -> Path:
    return WORKDIR / "outputs" / "mit_10m" / lang / f"captions_task{task_id:04d}.jsonl"


def tmp_output_path(lang: str, task_id: int) -> Path:
    return WORKDIR / "outputs" / "mit_10m" / lang / f"captions_task{task_id:04d}.jsonl.tmp"


def language_output_stats(lang: str) -> dict:
    output_dir = WORKDIR / "outputs" / "mit_10m" / lang
    finals = sorted(output_dir.glob("captions_task*.jsonl"))
    tmps = sorted(output_dir.glob("captions_task*.jsonl.tmp"))
    stats = {
        "final_count": len(finals),
        "tmp_count": len(tmps),
        "line_count": None,
    }
    if len(finals) == EXPECTED_FINAL_FILES and not tmps:
        total_lines = 0
        for path in finals:
            with path.open("r", encoding="utf-8") as handle:
                for _ in handle:
                    total_lines += 1
        stats["line_count"] = total_lines
    return stats


def language_complete(lang: str) -> bool:
    stats = language_output_stats(lang)
    return (
        stats["final_count"] == EXPECTED_FINAL_FILES
        and stats["tmp_count"] == 0
        and stats["line_count"] == EXPECTED_LINES
    )


def get_missing_task_ids(lang: str) -> list[int]:
    missing = []
    for task_id in range(LOGICAL_TASK_COUNT):
        if not final_output_path(lang, task_id).exists():
            missing.append(task_id)
    return missing


def array_slot_log_path(lang: str, job_id: int | str, array_id: int) -> Path:
    return LOG_DIR / f"recaption-mit-10m-{lang}_{job_id}_{array_id}.out"


def slot_task_ids(array_id: int) -> list[int]:
    start = array_id * WORKERS_PER_NODE
    return list(range(start, start + WORKERS_PER_NODE))


def current_full_job_id(lang: str, state: dict) -> int | None:
    job_id = state.get("latest_full_jobs", {}).get(lang)
    return int(job_id) if job_id else None


def current_repair_job_id(lang: str, state: dict) -> int | None:
    job_id = state.get("latest_repair_jobs", {}).get(lang)
    return int(job_id) if job_id else None


def submit_full_job(lang: str) -> int:
    output = run_cmd(
        [
            "sbatch",
            "--parsable",
            "--job-name",
            f"recaption-mit-10m-{lang}",
            "--array",
            "0-15",
            "--export",
            f"ALL,RECAPTION_LOADER={LOADER_MODULE},RECAPTION_TARGET_LANG={lang}",
            str(RUN_SCRIPT),
        ]
    )
    return int(output.strip())


def chunked(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def submit_repair_job(lang: str, task_ids: list[int]) -> int:
    workers = len(task_ids)
    output = run_cmd(
        [
            "sbatch",
            "--parsable",
            "--job-name",
            f"recaption-mit-10m-{lang}-repair",
            "--nodes",
            "1",
            "--ntasks",
            "1",
            "--gpus-per-task",
            str(workers),
            "--cpus-per-task",
            str(workers * CPUS_PER_WORKER),
            "--export",
            (
                "ALL,"
                f"RECAPTION_LOADER={LOADER_MODULE},"
                f"RECAPTION_TARGET_LANG={lang},"
                f"RECAPTION_EXPLICIT_TASK_IDS={':'.join(str(task_id) for task_id in task_ids)},"
                f"RECAPTION_LOGICAL_TASK_COUNT={LOGICAL_TASK_COUNT}"
            ),
            str(REPAIR_SCRIPT),
        ]
    )
    return int(output.strip())


def cancel_job(job_id: str) -> None:
    run_cmd(["scancel", job_id], check=False)


def slot_maybe_hung(
    lang: str,
    job_id: int,
    array_id: int,
    parsed: dict,
    state: dict,
) -> bool:
    log_path = array_slot_log_path(lang, job_id, array_id)
    if not log_path.exists():
        return False

    missing = [task_id for task_id in slot_task_ids(array_id) if not final_output_path(lang, task_id).exists()]
    if not missing:
        return False

    age_seconds = time.time() - log_path.stat().st_mtime
    if age_seconds < HANG_MINUTES * 60:
        return False

    if array_id not in parsed["running_slots"].get(str(job_id), set()):
        return False

    suspect_key = f"{lang}:{job_id}:{array_id}"
    suspected = state.setdefault("suspected_slots", {})
    current_count = int(suspected.get(suspect_key, 0)) + 1
    suspected[suspect_key] = current_count
    if current_count < SUSPECT_POLLS_REQUIRED:
        log(
            f"Suspecting hung slot {job_id}_{array_id} for {lang}: "
            f"log idle for {math.floor(age_seconds / 60)}m, missing tasks={missing}"
        )
        return False

    return True


def clear_suspicions_for_live_slots(parsed: dict, state: dict) -> None:
    suspected = state.setdefault("suspected_slots", {})
    live_keys = set()
    for lang in ALL_LANGS:
        job_id = current_full_job_id(lang, state)
        if not job_id:
            continue
        running_slots = parsed["running_slots"].get(str(job_id), set())
        for array_id in running_slots:
            live_keys.add(f"{lang}:{job_id}:{array_id}")
    for key in list(suspected):
        if key not in live_keys:
            suspected.pop(key, None)


def mark_wave_manual_release(lang: str, state: dict) -> None:
    wave_index = LANG_TO_WAVE[lang]
    manual = state.setdefault("wave_requires_manual_release", {})
    manual[str(wave_index)] = True


def handle_stuck_slots(parsed: dict, state: dict) -> None:
    for lang in ALL_LANGS:
        job_id = current_full_job_id(lang, state)
        if not job_id:
            continue
        for array_id in sorted(parsed["running_slots"].get(str(job_id), set())):
            if not slot_maybe_hung(lang, job_id, array_id, parsed, state):
                continue

            missing = [task_id for task_id in slot_task_ids(array_id) if not final_output_path(lang, task_id).exists()]
            if not missing:
                continue

            log(f"Cancelling hung slot {job_id}_{array_id} for {lang}; missing tasks={missing}")
            cancel_job(f"{job_id}_{array_id}")
            mark_wave_manual_release(lang, state)
            repair_job_id = submit_repair_job(lang, missing)
            state.setdefault("latest_repair_jobs", {})[lang] = repair_job_id
            log(
                f"Submitted repair job {repair_job_id} for {lang} "
                f"with logical tasks {':'.join(str(task_id) for task_id in missing)}"
            )
            state.setdefault("suspected_slots", {}).pop(f"{lang}:{job_id}:{array_id}", None)


def repair_incomplete_languages(parsed: dict, state: dict) -> None:
    for lang in ALL_LANGS:
        if language_complete(lang):
            continue

        missing = get_missing_task_ids(lang)
        if not missing:
            continue

        full_job_id = current_full_job_id(lang, state)
        repair_job_id = current_repair_job_id(lang, state)

        full_job_active = False
        if full_job_id:
            full_states = parsed["base_states"].get(str(full_job_id), set())
            full_job_active = bool(full_states)
            if not full_job_active:
                full_job_states = parse_sacct_states(str(full_job_id))
                full_job_active = any(
                    state_name.startswith(("PENDING", "RUNNING", "COMPLETING", "CONFIGURING"))
                    for state_name in full_job_states
                )
                if any(is_failure_state(state_name) for state_name in full_job_states):
                    mark_wave_manual_release(lang, state)

        repair_job_active = False
        if repair_job_id:
            repair_states = parsed["base_states"].get(str(repair_job_id), set())
            repair_job_active = bool(repair_states)
            if not repair_job_active:
                repair_job_states = parse_sacct_states(str(repair_job_id))
                repair_job_active = any(
                    state_name.startswith(("PENDING", "RUNNING", "COMPLETING", "CONFIGURING"))
                    for state_name in repair_job_states
                )

        if full_job_active or repair_job_active:
            continue

        mark_wave_manual_release(lang, state)
        submitted = []
        for chunk in chunked(missing, WORKERS_PER_NODE):
            repair_job_id = submit_repair_job(lang, chunk)
            submitted.append((repair_job_id, chunk))
            state.setdefault("latest_repair_jobs", {})[lang] = repair_job_id
        if submitted:
            joined = ", ".join(f"{job_id}:{':'.join(str(task_id) for task_id in chunk)}" for job_id, chunk in submitted)
            log(f"Submitted repair jobs for {lang}: {joined}")


def maybe_release_next_wave(parsed: dict, state: dict) -> None:
    manual_release = state.setdefault("wave_requires_manual_release", {})
    for wave_index, wave in enumerate(WAVES, start=1):
        if not all(language_complete(lang) for lang in wave):
            continue

        if wave_index >= len(WAVES):
            continue

        if not manual_release.get(str(wave_index)):
            continue

        next_wave = WAVES[wave_index]
        log(f"Wave {wave_index} is complete after repairs; manually releasing wave {wave_index + 1}")
        for lang in next_wave:
            job_id = current_full_job_id(lang, state)
            active_entries = []
            if job_id:
                active_entries = [
                    entry for entry in parsed["entries"] if entry["job_id"].split("_", 1)[0] == str(job_id)
                ]
            if active_entries and any(entry["state"] == "R" for entry in active_entries):
                continue
            if active_entries and all(entry["state"] == "PD" for entry in active_entries):
                log(f"Cancelling blocked pending job {job_id} for {lang} before resubmission")
                cancel_job(str(job_id))
            new_job_id = submit_full_job(lang)
            state.setdefault("latest_full_jobs", {})[lang] = new_job_id
            log(f"Submitted full job {new_job_id} for {lang}")
        manual_release[str(wave_index)] = False


def maybe_start_uno(state: dict) -> None:
    if state.get("uno_started"):
        return
    if not UNO_SCRIPT.exists():
        return
    if not all(language_complete(lang) for lang in ALL_LANGS):
        return

    subprocess.Popen(
        [sys.executable, str(UNO_SCRIPT)],
        cwd=WORKDIR,
        stdout=(LOG_DIR / "uno_stage2_prepare.out").open("a", encoding="utf-8"),
        stderr=(LOG_DIR / "uno_stage2_prepare.err").open("a", encoding="utf-8"),
        start_new_session=True,
    )
    state["uno_started"] = True
    log(f"MIT-10M is complete; started UNO stage-2 preparation via {UNO_SCRIPT.name}")


def run_iteration(state: dict) -> None:
    parsed = enrich_squeue(parse_squeue())
    discover_jobs(parsed, state)
    clear_suspicions_for_live_slots(parsed, state)
    handle_stuck_slots(parsed, state)
    repair_incomplete_languages(parsed, state)
    maybe_release_next_wave(parsed, state)
    maybe_start_uno(state)
    save_state(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = load_state()
    log("Starting MIT-10M watcher")
    while True:
        try:
            run_iteration(state)
        except Exception as exc:  # pragma: no cover - safety loop
            log(f"Watcher iteration failed: {exc!r}")
            save_state(state)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
