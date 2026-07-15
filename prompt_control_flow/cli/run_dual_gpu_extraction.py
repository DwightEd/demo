from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO


VALID_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
VALID_MODES = ("selected", "geometry")
LONG_FIRST_SUBSETS = ("omnimath", "olympiadbench", "math", "gsm8k")


@dataclass(frozen=True)
class ExtractionJob:
    subset: str
    mode: str
    output_path: Path

    @property
    def name(self) -> str:
        return f"{self.mode}:{self.subset}"


@dataclass(frozen=True)
class ControllerConfig:
    input_path: Path
    model_path: Path
    output_root: Path
    extraction_script: Path
    layers: str = "8,10,12,14,16,18,20,22"
    replay_protocol: str = "processbench_observer_chat_v1"
    max_seq_len: int = 4096
    min_success_fraction: float = 0.95
    max_chains: int = 0
    state_storage_dtype: str = "float16"
    trust_remote_code: bool = False


@dataclass
class RunningJob:
    job: ExtractionJob
    gpu: str
    process: subprocess.Popen[bytes]
    log_path: Path
    log_stream: TextIO
    started_at: float
    command: list[str]


def parse_csv(value: str, *, name: str, unique: bool = False) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if unique and len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values: {values}")
    return values


def build_jobs(
    subsets: Sequence[str],
    modes: Sequence[str],
    output_root: Path,
) -> list[ExtractionJob]:
    subset_set = {str(value) for value in subsets}
    mode_values = tuple(str(value) for value in modes)
    unknown_subsets = subset_set.difference(VALID_SUBSETS)
    unknown_modes = set(mode_values).difference(VALID_MODES)
    if unknown_subsets:
        raise ValueError(f"unknown ProcessBench subset(s): {sorted(unknown_subsets)}")
    if unknown_modes:
        raise ValueError(f"unknown extraction mode(s): {sorted(unknown_modes)}")

    ordered_subsets = [name for name in LONG_FIRST_SUBSETS if name in subset_set]
    return [
        ExtractionJob(
            subset=subset,
            mode=mode,
            output_path=Path(output_root) / subset / mode / "trace.npz",
        )
        for mode in mode_values
        for subset in ordered_subsets
    ]


def build_extraction_command(
    job: ExtractionJob,
    config: ControllerConfig,
    *,
    python_executable: str,
) -> list[str]:
    command = [
        str(python_executable),
        str(config.extraction_script),
        "--input",
        str(config.input_path),
        "--input_format",
        "processbench_source",
        "--subset",
        job.subset,
        "--model",
        str(config.model_path),
        "--output",
        str(job.output_path),
        "--replay_protocol",
        str(config.replay_protocol),
        "--max_seq_len",
        str(int(config.max_seq_len)),
        "--min_success_fraction",
        str(float(config.min_success_fraction)),
        "--dtype",
        "bfloat16",
        "--device",
        "cuda",
        "--state_storage_dtype",
        str(config.state_storage_dtype),
    ]
    if int(config.max_chains) > 0:
        command.extend(("--max_chains", str(int(config.max_chains))))
    if config.trust_remote_code:
        command.append("--trust_remote_code")

    if job.mode == "selected":
        command.extend(
            (
                "--layers",
                str(config.layers),
                "--enable_prompt_flow",
                "--enable_uncertainty",
                "--store_step_vectors",
                "--store_step_state_vectors",
                "--store_prompt_token_states",
                "--store_response_token_states",
            )
        )
    elif job.mode == "geometry":
        command.append("--geometry_only")
    else:  # pragma: no cover - build_jobs validates this contract
        raise ValueError(f"unsupported extraction mode {job.mode!r}")
    return command


def _controller_environment(gpu: str, cpu_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": str(max(int(cpu_threads), 1)),
            "MKL_NUM_THREADS": str(max(int(cpu_threads), 1)),
        }
    )
    return env


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.json")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, ensure_ascii=False)
    partial.replace(path)


def run_jobs(
    jobs: Sequence[ExtractionJob],
    config: ControllerConfig,
    *,
    gpus: Sequence[str],
    python_executable: str,
    cpu_threads_per_worker: int,
    poll_seconds: float,
    resume: bool,
    dry_run: bool,
) -> int:
    available = deque(str(gpu) for gpu in gpus)
    if not available:
        raise ValueError("at least one GPU is required")

    pending: deque[ExtractionJob] = deque()
    summary_rows: list[dict[str, object]] = []
    for job in jobs:
        command = build_extraction_command(
            job, config, python_executable=python_executable
        )
        if job.output_path.exists():
            if not resume:
                raise FileExistsError(
                    f"{job.output_path} already exists; use --resume to skip it"
                )
            summary_rows.append(
                {
                    "job": job.name,
                    "status": "skipped_existing",
                    "output": str(job.output_path),
                    "command": command,
                }
            )
            continue
        pending.append(job)

    if dry_run:
        for index, job in enumerate(pending):
            gpu = tuple(available)[index % len(available)]
            command = build_extraction_command(
                job, config, python_executable=python_executable
            )
            print(
                f"CUDA_VISIBLE_DEVICES={gpu} "
                f"{shlex.join(command)}"
            )
        return 0

    config.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = config.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[int, RunningJob] = {}

    def launch(job: ExtractionJob, gpu: str) -> None:
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job.mode}_{job.subset}.log"
        log_stream = log_path.open("w", encoding="utf-8")
        command = build_extraction_command(
            job, config, python_executable=python_executable
        )
        print(f"[start] gpu={gpu} job={job.name} log={log_path}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=str(config.extraction_script.resolve().parent),
            env=_controller_environment(gpu, cpu_threads_per_worker),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        running[int(process.pid)] = RunningJob(
            job=job,
            gpu=gpu,
            process=process,
            log_path=log_path,
            log_stream=log_stream,
            started_at=time.monotonic(),
            command=command,
        )

    try:
        while pending or running:
            while pending and available:
                launch(pending.popleft(), available.popleft())

            completed = [pid for pid, item in running.items() if item.process.poll() is not None]
            if not completed:
                time.sleep(max(float(poll_seconds), 0.05))
                continue

            for pid in completed:
                item = running.pop(pid)
                return_code = int(item.process.returncode or 0)
                elapsed = time.monotonic() - item.started_at
                item.log_stream.close()
                status = "completed" if return_code == 0 else "failed"
                summary_rows.append(
                    {
                        "job": item.job.name,
                        "status": status,
                        "gpu": item.gpu,
                        "return_code": return_code,
                        "elapsed_seconds": elapsed,
                        "output": str(item.job.output_path),
                        "log": str(item.log_path),
                        "command": item.command,
                    }
                )
                print(
                    f"[{status}] gpu={item.gpu} job={item.job.name} "
                    f"elapsed={elapsed / 60.0:.1f}m",
                    flush=True,
                )
                if return_code != 0:
                    for active in running.values():
                        active.process.terminate()
                    for active in running.values():
                        active.process.wait()
                        active.log_stream.close()
                    _write_summary(
                        config.output_root / "controller_summary.json", summary_rows
                    )
                    return return_code
                available.append(item.gpu)
    except BaseException:
        for active in running.values():
            active.process.terminate()
        for active in running.values():
            active.process.wait()
            active.log_stream.close()
        raise

    _write_summary(config.output_root / "controller_summary.json", summary_rows)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Run independent ProcessBench extraction jobs across multiple GPUs. "
            "Each worker owns one complete model replica and one complete output artifact."
        )
    )
    parser.add_argument("--input", default="data/hf_datasets/ProcessBench")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output_root", default="data/exact/processbench_observer_llama31"
    )
    parser.add_argument("--subsets", default=",".join(VALID_SUBSETS))
    parser.add_argument("--modes", default=",".join(VALID_MODES))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--layers", default="8,10,12,14,16,18,20,22")
    parser.add_argument(
        "--replay_protocol", default="processbench_observer_chat_v1"
    )
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--min_success_fraction", type=float, default=0.95)
    parser.add_argument("--max_chains", type=int, default=0)
    parser.add_argument(
        "--state_storage_dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--cpu_threads_per_worker", type=int, default=4)
    parser.add_argument("--poll_seconds", type=float, default=1.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--extraction_script", default=str(repo_root / "extract_mechanisms.py")
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    try:
        subsets = parse_csv(args.subsets, name="subsets", unique=True)
        modes = parse_csv(args.modes, name="modes", unique=True)
        gpus = parse_csv(args.gpus, name="gpus", unique=True)
        config = ControllerConfig(
            input_path=Path(args.input).resolve(),
            model_path=Path(args.model).resolve(),
            output_root=Path(args.output_root).resolve(),
            extraction_script=Path(args.extraction_script).resolve(),
            layers=str(args.layers),
            replay_protocol=str(args.replay_protocol),
            max_seq_len=int(args.max_seq_len),
            min_success_fraction=float(args.min_success_fraction),
            max_chains=int(args.max_chains),
            state_storage_dtype=str(args.state_storage_dtype),
            trust_remote_code=bool(args.trust_remote_code),
        )
        jobs = build_jobs(subsets, modes, config.output_root)
        print(
            f"jobs={len(jobs)} gpus={list(gpus)} model={config.model_path}",
            flush=True,
        )
        exit_code = run_jobs(
            jobs,
            config,
            gpus=gpus,
            python_executable=str(args.python),
            cpu_threads_per_worker=int(args.cpu_threads_per_worker),
            poll_seconds=float(args.poll_seconds),
            resume=bool(args.resume),
            dry_run=bool(args.dry_run),
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
