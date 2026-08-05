from __future__ import annotations

from pathlib import Path


def test_exact_modes_require_verified_raw_residual_manifest() -> None:
    runner = Path(__file__).resolve().parents[1] / "run_raw_remote.sh"
    script = runner.read_text(encoding="utf-8")

    assert (
        'EXACT_MANIFEST_NAME="${EXACT_MANIFEST_NAME:-trace.raw_residual_stream.npz}"'
        in script
    )
    assert 'manifest="${data_root}/${subset}/selected/${EXACT_MANIFEST_NAME}"' in script
    assert 'manifest="${data_root}/${subset}/selected/trace.npz"' not in script


def test_exact_pilot_samples_pairs_from_llama_filtered_full_manifests() -> None:
    runner = Path(__file__).resolve().parents[1] / "run_raw_remote.sh"
    script = runner.read_text(encoding="utf-8")

    assert 'data_root="${REPO_ROOT}/data/exact/processbench_observer_llama31_full"' in script
    assert "--response-generator llama3.1-8b" in script
    assert "extra=(--max-pairs 20" in script


def test_remote_runner_checks_the_active_python_environment_and_runs_foreground() -> None:
    runner = Path(__file__).resolve().parents[1] / "run_raw_remote.sh"
    script = runner.read_text(encoding="utf-8")

    assert '"${PYTHON_BIN}" -c' in script
    assert "import sklearn" in script
    assert "sys.executable" in script
    assert "sklearn.__file__" in script
    assert "PYTHONUNBUFFERED=1" in script
    assert "screen -dmS" not in script


def test_hidden_geometry_runner_has_causal_first_error_modes() -> None:
    runner = Path(__file__).resolve().parents[1] / "run_hidden_geometry_remote.sh"
    script = runner.read_text(encoding="utf-8")

    assert (
        'MODEL_DIR="${MODEL_DIR:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"'
        in script
    )
    assert 'GPU_ID="${GPU_ID:-0}"' in script
    assert 'CAUSAL_LAYERS="${CAUSAL_LAYERS:-8,12,16,20,24,28}"' in script
    assert 'CAUSAL_DOMAINS="${CAUSAL_DOMAINS:-gsm8k,math,olympiadbench,omnimath}"' in script
    assert (
        'DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/research/demo/data/exact/processbench_observer_llama31_full}"'
        in script
    )
    assert '--domains "${CAUSAL_DOMAINS}"' in script
    assert "export CUDA_VISIBLE_DEVICES" in script
    for mode in (
        "causal-audit",
        "causal-extract-smoke",
        "causal-intervene-smoke",
        "causal-summarize-smoke",
        "causal-monitor-smoke",
        "causal-monitor-full",
        "causal-full",
    ):
        assert f"  {mode})" in script
    assert "causal_first_error_attribution.main extract" in script
    assert "causal_first_error_attribution.main intervene" in script
    assert "causal_first_error_attribution.main summarize" in script
    assert "causal_first_error_attribution.main train-monitor" in script
    assert 'geometry_trace="${DATA_ROOT}/${domain}/geometry/trace.npz"' in script
    assert 'if [[ "${MODE}" == causal-monitor-* && ! -f "${geometry_trace}" ]]; then' in script
    assert "MONITOR_SCORES" not in script
    assert "component-smoke)" not in script
    assert "component-full)" not in script
    assert "--method component_resolved_hazard" not in script
    assert "source activate" not in script
    assert "conda activate" not in script
    assert (
        'if [[ "${MODE}" != causal-monitor-* && ! -f "${aligned_trace}" ]]; then'
        in script
    )
    assert "--shuffle-repeats 2" in script
    assert "--shuffle-repeats 3" in script


def test_hidden_geometry_runner_audits_causal_pairs_without_loading_model() -> None:
    runner = Path(__file__).resolve().parents[1] / "run_hidden_geometry_remote.sh"
    script = runner.read_text(encoding="utf-8")

    marker = "  causal-audit)"
    assert marker in script
    start = script.index(marker)
    branch = script[start : script.index("    ;;", start)]
    assert "causal_first_error_attribution.main audit" in branch
    assert "require_component_runtime" not in branch
    assert '--output "${OUTPUT_ROOT}/causal_audit_${RUN_TAG}.json"' in branch
    assert "nohup" not in script
    assert "screen -dmS" not in script
    assert "tmux" not in script
