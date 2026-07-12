import numpy as np

from anchorflow.volume import (
    anchor_residual_cloud,
    conditional_gram_geometry,
    gram_features,
    gram_spectrum,
)


def test_gram_geometry_separates_rank_one_from_diffuse_cloud():
    rank_one = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (6, 1))
    diffuse = np.eye(4)
    low = gram_features(rank_one)
    high = gram_features(diffuse)
    assert low["eff_rank"] < 1.01
    assert high["eff_rank"] > 3.9
    assert high["tail_auc"] > low["tail_auc"]
    np.testing.assert_allclose(gram_spectrum(rank_one), [1.0], atol=1e-10)


def test_conditional_gram_features_are_explicitly_gated():
    rank_one = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (6, 1))
    diffuse = np.eye(4)
    out = conditional_gram_geometry([rank_one, diffuse], condition=[False, True])
    assert np.isnan(out["conditional_eff_rank"][0])
    assert np.isfinite(out["conditional_eff_rank"][1])
    assert out["condition_active"].tolist() == [0.0, 1.0]
    assert out["spectral_js_change"][1] > 0.0


def test_anchor_conditioning_projects_prompt_span_component_out():
    cloud = np.array([[1.0, 1.0], [1.0, -1.0]])
    anchors = np.array([[1.0, 0.0]])
    residual, ratio, rank = anchor_residual_cloud(cloud, anchors)
    np.testing.assert_allclose(residual[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(ratio, 0.5, atol=1e-12)
    assert rank == 1

    out = conditional_gram_geometry(
        [cloud],
        anchor_vectors=anchors,
        condition=[True],
    )
    np.testing.assert_allclose(out["residual_energy_ratio"], [0.5], atol=1e-12)
    np.testing.assert_allclose(out["residual_eff_rank"], [1.0], atol=1e-12)
    assert np.isfinite(out["conditional_residual_tail_auc"][0])
