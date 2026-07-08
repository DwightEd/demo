import argparse
import os

import trajectory_phase_transition_audit as tpt


def _args(tmp_path, npz):
    return argparse.Namespace(
        input=str(npz),
        layer=16,
        nearest_layer=True,
        layer_is_id=False,
        signals="spread,entropy_mean",
        min_prefix=2,
        pre_window=3,
        post_window=3,
        event_q=0.90,
        stable_q=0.50,
        scale_floor=1e-3,
        std_floor_frac=0.05,
        bootstrap=10,
        seed=5,
        max_chains=0,
        output_dir=str(tmp_path),
        no_progress=True,
        selftest=False,
    )


def test_phase_transition_selftest_runs_and_writes_outputs(tmp_path):
    npz = tmp_path / "trajectory_phase_transition_selftest.npz"
    tpt.make_selftest(str(npz), seed=3)
    res = tpt.run(str(npz), _args(tmp_path, npz))
    tpt.assert_selftest(res)

    spread = res["signals"]["spread"]
    assert spread["event_detection"]["break_z"]["first_error_auroc"] >= 0.95
    assert spread["response_detection"]["max_break_z"]["chain_auroc"] >= 0.95
    assert spread["gold_event_ranks"]["break_z"]["top1_rate"] >= 0.80
    assert "stable_prefix_break" in spread["first_error_modes"]["modes"]

    paths = tpt.write_outputs(res, str(tmp_path), "trajectory_phase_transition_selftest")
    for path in paths:
        assert os.path.exists(path)
