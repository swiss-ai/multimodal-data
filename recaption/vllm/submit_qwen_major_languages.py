#!/usr/bin/env python3

from __future__ import annotations

import subprocess
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent.parent
RUN_SCRIPT = WORKDIR / "run.slurm"
LOADER_MODULE = "loader_mitsua_art_museums_pd_440k_qwen_from_en"
WAVES = [
    ["fr", "de"],
    ["pt", "it"],
    ["ru", "ar"],
    ["zh-hans", "zh-hant"],
    ["ja", "ko"],
    ["hi", "tr"],
    ["vi", "id"],
    ["th"],
]


def submit_job(lang: str, dependency: str | None) -> str:
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        f"recaption-art-museums-pd-440k-{lang}",
        "--array",
        "0-9",
    ]
    if dependency:
        cmd.extend(["--dependency", dependency])
    cmd.extend(
        [
            "--export",
            (f"ALL,RECAPTION_LOADER={LOADER_MODULE},RECAPTION_TARGET_LANG={lang}"),
            str(RUN_SCRIPT),
        ]
    )

    result = subprocess.run(
        cmd,
        cwd=WORKDIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    previous_wave_ids: list[str] = []
    for wave_index, wave in enumerate(WAVES, start=1):
        dependency = None
        if previous_wave_ids:
            dependency = f"afterok:{':'.join(previous_wave_ids)}"

        current_wave_ids: list[str] = []
        for lang in wave:
            job_id = submit_job(lang, dependency)
            current_wave_ids.append(job_id)
            print(
                f"wave={wave_index} lang={lang} job_id={job_id}" + (f" dependency={dependency}" if dependency else "")
            )

        previous_wave_ids = current_wave_ids


if __name__ == "__main__":
    main()
