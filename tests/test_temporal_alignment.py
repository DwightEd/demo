import numpy as np

from multisample_temporal_rupture_audit import (
    build_base_sequences,
    build_multichannel_scores,
    pool_token_trace_to_steps,
)


class Bundle(dict):
    @property
    def files(self):
        return list(self.keys())


def test_token_trace_pools_by_cloud_sizes_before_step_fusion():
    clouds = np.arange(10, dtype=np.float64).reshape(5, 1, 2)
    data = Bundle(
        problem_ids=np.array([7]),
        n_steps=np.array([2]),
        sv_clouds=np.asarray([clouds], dtype=object),
        cloud_sizes=np.asarray([np.array([2, 3])], dtype=object),
        sv_out_entropy=np.asarray([np.array([0.2, 0.4])], dtype=object),
        sv_tok_entropy=np.asarray([np.array([1.0, 3.0, 5.0, 7.0, 9.0])], dtype=object),
    )

    seq = build_base_sequences(data, bands=["all"])[0]

    np.testing.assert_allclose(seq["tok_entropy"], [2.0, 7.0])
    np.testing.assert_allclose(seq["out_entropy"], [0.2, 0.4])
    assert seq["tok_entropy"].shape == seq["cloud_spread"].shape == (2,)


def test_token_trace_uses_ranges_when_steps_have_token_gaps():
    values = np.arange(7, dtype=np.float64)
    ranges = np.array([[10, 11], [14, 16]])  # inclusive historical convention

    pooled = pool_token_trace_to_steps(
        values,
        cloud_sizes=np.array([2, 3]),
        step_ranges=ranges,
        expected_steps=2,
    )

    np.testing.assert_allclose(pooled, [0.5, 5.0])


def test_token_trace_without_boundaries_is_not_mistaken_for_step_data():
    # Equal lengths are not sufficient evidence that two arrays share an axis.
    assert pool_token_trace_to_steps(
        np.array([0.1, 0.2, 0.3]),
        expected_steps=3,
    ) is None


def test_multichannel_fusion_requires_exact_step_axis_not_min_length():
    exact = {
        "__step_count__": 4,
        "step_pos": np.linspace(0.0, 1.0, 4),
        "a": np.array([0.0, 1.0, 0.0, 2.0]),
        "b": np.array([1.0, 0.0, 2.0, 1.0]),
    }
    with_misaligned = dict(exact)
    with_misaligned["token_axis"] = np.array([100.0, 200.0])

    base = build_multichannel_scores([exact], ["a", "b"], width=1)
    got = build_multichannel_scores([with_misaligned], ["a", "b", "token_axis"], width=1)

    for name in base:
        np.testing.assert_allclose(got[name][0], base[name][0], equal_nan=True)
        np.testing.assert_allclose(got[name][1], base[name][1], equal_nan=True)

    only_mismatch = build_multichannel_scores(
        [{"__step_count__": 4, "a": exact["a"], "token_axis": np.array([1.0, 2.0])}],
        ["a", "token_axis"],
        width=1,
    )
    assert all(np.isnan(values[0][0]) for values in only_mismatch.values())
