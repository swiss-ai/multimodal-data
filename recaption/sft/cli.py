from __future__ import annotations

import os
import subprocess
import sys

import typer
from rich import print
from sft_recaption.config import (
    DEFAULT_TASK_COUNT,
    ModelConfig,
    ensure_runtime_dirs,
    resolve_model_download_dir,
    resolve_model_reference,
)
from sft_recaption.loaders import create_loader
from sft_recaption.pipeline import (
    export_hf_dataset,
    generate_candidates,
    get_pending_generation_records,
    judge_examples,
    prepare_manifests,
    set_cpu_affinity,
)
from sft_recaption.runtime import VLLMChatEngine, configure_worker_environment

app = typer.Typer(no_args_is_help=True)


def build_model_config(
    *,
    model_repo: str,
    tensor_parallel_size: int,
    max_num_seqs: int,
    enforce_eager: bool,
) -> ModelConfig:
    resolved_model_repo = resolve_model_reference(model_repo)
    return ModelConfig(
        model_repo=resolved_model_repo,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
        download_dir=resolve_model_download_dir(),
    )


def spawn_local_workers(
    *,
    loader_name: str,
    split: str,
    model_repo: str,
    batch_size: int,
    limit_per_worker: int | None,
    enforce_eager: bool,
    task_count: int,
    task_offset: int,
) -> None:
    children: list[subprocess.Popen[str]] = []
    try:
        for worker_index in range(4):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(worker_index)
            command = [
                sys.executable,
                "-m",
                "sft_recaption.cli",
                "generate",
                "--loader",
                loader_name,
                "--task-id",
                str(task_offset + worker_index),
                "--task-count",
                str(task_count),
                "--split",
                split,
                "--model",
                model_repo,
                "--batch-size",
                str(batch_size),
                *(["--limit", str(limit_per_worker)] if limit_per_worker is not None else []),
                "--max-num-seqs",
                str(batch_size),
                *(["--enforce-eager"] if enforce_eager else []),
                "--worker-index",
                str(worker_index),
            ]
            children.append(subprocess.Popen(command, env=env))

        exit_codes = [child.wait() for child in children]
        failures = [task_offset + worker_index for worker_index, code in enumerate(exit_codes) if code != 0]
        if failures:
            print(
                f"Workers failed for task ids {failures}; completed outputs are kept and reruns will resume missing work."
            )
        else:
            print(f"Generation completed across workers {task_offset}..{task_offset + 3}")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()


@app.command("prepare-manifests")
def prepare_manifests_cmd(
    loader_name: str = typer.Option(..., "--loader"),
    task_count: int = typer.Option(DEFAULT_TASK_COUNT, "--task-count"),
    split: str = typer.Option("train", "--split"),
) -> None:
    ensure_runtime_dirs()
    loader = create_loader(loader_name)
    paths = prepare_manifests(loader, task_count=task_count, split=split)
    print(f"Wrote {len(paths)} manifest shards under {paths[0].parent}")


@app.command("generate")
def generate_cmd(
    loader_name: str = typer.Option(..., "--loader"),
    task_id: int = typer.Option(..., "--task-id"),
    task_count: int = typer.Option(DEFAULT_TASK_COUNT, "--task-count"),
    split: str = typer.Option("train", "--split"),
    model_repo: str = typer.Option("google/gemma-4-26B-A4B-it", "--model"),
    batch_size: int = typer.Option(2, "--batch-size"),
    limit: int | None = typer.Option(None, "--limit"),
    max_num_seqs: int = typer.Option(2, "--max-num-seqs"),
    tensor_parallel_size: int = typer.Option(1, "--tensor-parallel-size"),
    enforce_eager: bool = typer.Option(False, "--enforce-eager"),
    worker_index: int = typer.Option(0, "--worker-index"),
) -> None:
    ensure_runtime_dirs()
    configure_worker_environment(worker_index)
    set_cpu_affinity(worker_index)
    loader = create_loader(loader_name)
    output_path, pending_records = get_pending_generation_records(
        loader,
        task_id=task_id,
        task_count=task_count,
        split=split,
        limit=limit,
    )
    if not pending_records:
        print(f"Skipping task {task_id}; candidates already complete at {output_path}")
        return
    config = build_model_config(
        model_repo=model_repo,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
    )
    if loader.limit_mm_per_prompt is not None:
        config.limit_mm_per_prompt = loader.limit_mm_per_prompt
    engine = VLLMChatEngine(config)
    path = generate_candidates(
        loader,
        engine,
        task_id=task_id,
        task_count=task_count,
        batch_size=batch_size,
        split=split,
        limit=limit,
        model_repo=model_repo,
    )
    print(f"Wrote candidates to {path}")


@app.command("curate")
def curate_cmd(
    loader_name: str = typer.Option(..., "--loader"),
    model_repo: str = typer.Option("google/gemma-4-26B-A4B-it", "--model"),
    batch_size: int = typer.Option(2, "--batch-size"),
    max_num_seqs: int = typer.Option(2, "--max-num-seqs"),
    tensor_parallel_size: int = typer.Option(1, "--tensor-parallel-size"),
    enforce_eager: bool = typer.Option(False, "--enforce-eager"),
) -> None:
    ensure_runtime_dirs()
    loader = create_loader(loader_name)
    engine = VLLMChatEngine(
        build_model_config(
            model_repo=model_repo,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
        )
    )
    path = judge_examples(
        loader,
        engine,
        batch_size=batch_size,
        model_repo=model_repo,
    )
    print(f"Wrote curated examples to {path}")


@app.command("export-parquet")
@app.command("export-hf")
def export_hf_cmd(
    loader_name: str = typer.Option(..., "--loader"),
    stage: str = typer.Option("candidates", "--stage"),
) -> None:
    ensure_runtime_dirs()
    loader = create_loader(loader_name)
    path = export_hf_dataset(loader, stage=stage)
    print(f"Exported parquet shards to {path}")


@app.command("probe")
def probe_cmd(
    loader_name: str = typer.Option(..., "--loader"),
    split: str = typer.Option("train", "--split"),
    count: int = typer.Option(2, "--count"),
) -> None:
    loader = create_loader(loader_name)
    dataset = loader.get_split_dataset(split)
    print(
        {
            "loader": loader.name,
            "split": split,
            "rows": len(dataset),
            "columns": list(dataset.column_names),
            "probe_indices": list(range(min(count, len(dataset)))),
        }
    )


@app.command("run-local-4gpu")
def run_local_4gpu_cmd(
    loader_name: str = typer.Option(..., "--loader"),
    split: str = typer.Option("train", "--split"),
    model_repo: str = typer.Option("google/gemma-4-26B-A4B-it", "--model"),
    batch_size: int = typer.Option(2, "--batch-size"),
    limit_per_worker: int | None = typer.Option(None, "--limit-per-worker"),
    enforce_eager: bool = typer.Option(False, "--enforce-eager"),
) -> None:
    ensure_runtime_dirs()
    task_count = 4
    loader = create_loader(loader_name)
    prepare_manifests(loader, task_count=task_count, split=split)
    spawn_local_workers(
        loader_name=loader_name,
        split=split,
        model_repo=model_repo,
        batch_size=batch_size,
        limit_per_worker=limit_per_worker,
        enforce_eager=enforce_eager,
        task_count=task_count,
        task_offset=0,
    )


@app.command("run-node-workers")
def run_node_workers_cmd(
    loader_name: str = typer.Option(..., "--loader"),
    task_offset: int = typer.Option(..., "--task-offset"),
    task_count: int = typer.Option(..., "--task-count"),
    split: str = typer.Option("train", "--split"),
    model_repo: str = typer.Option("google/gemma-4-26B-A4B-it", "--model"),
    batch_size: int = typer.Option(2, "--batch-size"),
    limit_per_worker: int | None = typer.Option(None, "--limit-per-worker"),
    enforce_eager: bool = typer.Option(False, "--enforce-eager"),
) -> None:
    ensure_runtime_dirs()
    spawn_local_workers(
        loader_name=loader_name,
        split=split,
        model_repo=model_repo,
        batch_size=batch_size,
        limit_per_worker=limit_per_worker,
        enforce_eager=enforce_eager,
        task_count=task_count,
        task_offset=task_offset,
    )


if __name__ == "__main__":
    app()
