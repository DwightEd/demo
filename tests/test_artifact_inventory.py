from __future__ import annotations

import json

import numpy as np

from prompt_control_flow.artifact_inventory import inventory_artifacts, write_inventory


def test_npz_inventory_reads_schema_without_tensor_loading(tmp_path) -> None:
    artifact = tmp_path / "same_problem.npz"
    np.savez_compressed(
        artifact,
        problem_ids=np.asarray([0, 0]),
        sample_idx=np.asarray([0, 1]),
        is_correct=np.asarray([1, 0]),
        sv_clouds=np.zeros((2, 3, 1, 5), dtype=np.float16),
        cloud_sizes=np.asarray([[1, 2], [1, 2]]),
    )
    rows = inventory_artifacts([tmp_path], include_reports=False)
    assert len(rows) == 1
    assert "same_problem_raw_token_hidden_clouds" in rows[0]["roles"]
    assert "same_problem_sampling_axis" in rows[0]["roles"]
    paths = write_inventory(rows, tmp_path / "inventory")
    loaded = json.loads((tmp_path / "inventory" / "artifact_inventory.json").read_text())
    assert loaded[0]["path"].endswith("same_problem.npz")
    assert set(paths) == {"json", "csv", "markdown"}
