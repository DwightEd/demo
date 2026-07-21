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
