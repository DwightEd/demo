from __future__ import annotations

from pathlib import Path

from prompt_control_flow.cli.run_dual_gpu_extraction import (
    ControllerConfig,
    build_extraction_command,
    build_jobs,
    parse_csv,
)


def test_build_jobs_uses_long_first_order_and_isolated_outputs(tmp_path: Path) -> None:
    jobs = build_jobs(
        subsets=("gsm8k", "math", "olympiadbench", "omnimath"),
        modes=("selected", "geometry"),
        output_root=tmp_path,
    )

    assert [(job.mode, job.subset) for job in jobs[:4]] == [
        ("selected", "omnimath"),
        ("selected", "olympiadbench"),
        ("selected", "math"),
        ("selected", "gsm8k"),
    ]
    assert len({job.output_path for job in jobs}) == 8
    assert jobs[0].output_path == tmp_path / "omnimath" / "selected" / "trace.npz"
    assert jobs[4].output_path == tmp_path / "omnimath" / "geometry" / "trace.npz"


def test_selected_command_stores_compact_scores_and_selected_token_states(
    tmp_path: Path,
) -> None:
    config = ControllerConfig(
        input_path=Path("data/hf_datasets/ProcessBench"),
        model_path=Path("/models/Meta-Llama-3.1-8B-Instruct"),
        output_root=tmp_path,
        extraction_script=Path("extract_mechanisms.py"),
        layers="8,10,12,14,16,18,20,22",
        max_seq_len=4096,
        min_success_fraction=0.95,
        max_chains=0,
        state_storage_dtype="float16",
        trust_remote_code=False,
    )
    job = build_jobs(("gsm8k",), ("selected",), tmp_path)[0]

    command = build_extraction_command(job, config, python_executable="python")

    assert command[:2] == ["python", "extract_mechanisms.py"]
    assert command[command.index("--subset") + 1] == "gsm8k"
    assert "--store_step_vectors" in command
    assert "--store_step_state_vectors" in command
    assert "--store_prompt_token_states" in command
    assert "--store_response_token_states" in command
    assert "--enable_uncertainty" in command
    assert "--enable_icr" not in command
    assert "--geometry_only" not in command
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--device") + 1] == "cuda"


def test_geometry_command_uses_all_layers_without_logits_or_token_shards(
    tmp_path: Path,
) -> None:
    config = ControllerConfig(
        input_path=Path("data/hf_datasets/ProcessBench"),
        model_path=Path("/models/Meta-Llama-3.1-8B-Instruct"),
        output_root=tmp_path,
        extraction_script=Path("extract_mechanisms.py"),
    )
    job = build_jobs(("math",), ("geometry",), tmp_path)[0]

    command = build_extraction_command(job, config, python_executable="python")

    assert "--geometry_only" in command
    assert "--store_prompt_token_states" not in command
    assert "--store_response_token_states" not in command
    assert "--enable_uncertainty" not in command
    assert "--enable_icr" not in command


def test_parse_csv_rejects_duplicate_gpu_assignments() -> None:
    assert parse_csv("0,1", name="gpus", unique=True) == ("0", "1")

    try:
        parse_csv("0,0", name="gpus", unique=True)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:  # pragma: no cover - explicit assertion message
        raise AssertionError("duplicate GPU IDs must be rejected")
