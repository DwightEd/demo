import argparse
import os

import trajectory_phase_transition_audit as tpt
import trajectory_signal_visualization as tsv


def _args(tmp_path, npz):
    return argparse.Namespace(
        input=str(npz),
        layer=16,
        nearest_layer=True,
        layer_is_id=False,
        signals="spread,entropy_mean",
        event_metrics="break_z,jump_z",
        min_prefix=2,
        event_q=0.90,
        stable_q=0.50,
        scale_floor=1e-3,
        std_floor_frac=0.05,
        topk=2,
        lse_tau=1.0,
        sort_by="break",
        norm_bins=16,
        align_window=2,
        case_count=4,
        plot_all_cases=False,
        max_chains=0,
        output_dir=str(tmp_path),
        no_progress=True,
        seed=7,
        bootstrap=0,
        dpi=80,
        max_fig_width=8.0,
        max_fig_height=10.0,
        vmin_q=0.02,
        vmax_q=0.98,
        signal_cmap="viridis",
        event_cmap="magma",
        max_score_bars=8,
        selftest=False,
    )


def test_trajectory_signal_visualization_selftest_outputs(tmp_path):
    npz = tmp_path / "trajectory_signal_visualization_selftest.npz"
    tpt.make_selftest(str(npz), seed=6)
    res = tsv.run_visualization(str(npz), _args(tmp_path, npz))

    assert os.path.exists(res["html"])
    assert os.path.exists(res["json"])
    assert res["figures"]
    assert res["case_figures"]
    for path in res["figures"][:4] + res["case_figures"][:2]:
        assert os.path.exists(path)

    spread = res["signals"]["spread"]
    assert os.path.exists(spread["chain_csv"])
    scores = {row["score"]: row["chain_auroc"] for row in spread["response_scores"]}
    assert scores["max_break_z"] >= 0.95
