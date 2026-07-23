from __future__ import annotations

from functional_divergence.output import versioned_paths


def test_versioning_archives_old_latest_before_returning_write_target(tmp_path):
    latest = tmp_path / "results.json"
    latest.write_text("old", encoding="utf-8")

    targets = versioned_paths(latest)

    archives = list(tmp_path.glob("results_*.json"))
    assert targets == (latest,)
    assert len(archives) == 1
    assert archives[0].read_text(encoding="utf-8") == "old"
    latest.write_text("new", encoding="utf-8")
    assert archives[0].read_text(encoding="utf-8") == "old"
