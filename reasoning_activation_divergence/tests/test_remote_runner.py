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


def test_hidden_geometry_runner_has_component_extraction_modes() -> None:
    runner = Path(__file__).resolve().parents[1] / "run_hidden_geometry_remote.sh"
    script = runner.read_text(encoding="utf-8")

    assert (
        'MODEL_DIR="${MODEL_DIR:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"'
        in script
    )
    assert 'GPU_ID="${GPU_ID:-0}"' in script
    assert 'COMPONENT_LAYERS="${COMPONENT_LAYERS:-' in script
    assert 'COMPONENT_LAYERS="${COMPONENT_LAYERS:-8,12,16,20,24,28}"' in script
    assert 'DATA_ROOT="${DATA_ROOT:-' in script
    assert "export CUDA_VISIBLE_DEVICES" in script
    assert "component-smoke)" in script
    assert "component-full)" in script
    assert "-m functional_divergence.hidden_state_geometry.component_extract_cli" in script
    assert "--max-records-per-domain 32" in script
    assert "--method component_resolved_hazard" in script
    assert "--tasks post_step" in script
    assert "--bootstrap 2000" in script
    assert "source activate" not in script
    assert "conda activate" not in script
