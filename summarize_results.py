"""Collate all analysis .npz outputs in data/ into one readable summary.

The per-script .npz (data/decomp_*, frac_*, probe_*, norm_*, temporal_*,
sparse_*, within_*) are machine-readable arrays saved for plotting; they are
gitignored and live only on the machine that ran them. This script loads
whatever is present and prints + writes a single human-readable
`results_summary.md` so you can see every result in one place.

Usage:  python summarize_results.py            # globs data/*.npz
        python summarize_results.py --glob 'data/decomp_*.npz'
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def _f(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "nan"


def _pairs(arr):
    """Robustly coerce a saved curve (list of (within,cross)) into an (n,2) float
    array, regardless of whether numpy stored it as (n,2), flattened, or an
    object array of tuples."""
    flat = []
    for x in np.asarray(arr, dtype=object).ravel():
        try:
            flat.extend(float(v) for v in x)
        except TypeError:
            flat.append(float(x))
    a = np.asarray(flat, dtype=float)
    return a.reshape(-1, 2) if a.size % 2 == 0 else a.reshape(-1, 1)


def summarize(path):
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        return f"## {os.path.basename(path)}\n  (could not load: {e})\n"
    k = set(d.files)
    name = os.path.basename(path)
    out = [f"## {name}"]

    if "probe_within_auroc" in k:                                   # 12
        out.append("**learned probe + difficulty inflation**")
        out.append(f"- within-problem probe (HONEST) = {_f(d['probe_within_auroc'])}"
                   f" +/- {_f(d.get('probe_within_std', 0))}")
        out.append(f"- GROUP-kfold pooled (held-out probs, pooled metric) = "
                   f"{_f(d.get('probe_group_pooled','nan'))}  <- same held-out preds as within, pooled")
        out.append(f"- cross-problem pooled (random split, INFLATED) = {_f(d.get('probe_cross_pooled','nan'))}")
        out.append(f"- length baseline = {_f(d.get('length_within_auroc','nan'))}"
                   f"   unsupervised = {_f(d.get('base_within_auroc','nan'))}")

    elif "cosine" in k:                                             # 15
        out.append("**difficulty vs failure directions**")
        out.append(f"- cosine(w_fail,w_diff) = {_f(d['cosine'])};  "
                   f"difficulty held-out AUROC = {_f(d['diff_auroc'])}")
        out.append(f"- failure raw = {_f(d['fail_auroc_raw'])};  "
                   f"difficulty-removed = {_f(d['fail_auroc_diffout'])}")

    elif "curve" in k and "n_bins" in k:                            # 16
        out.append("**fractional-position emergence (within | cross)**")
        nb = int(d["n_bins"])
        for b, (w, c) in enumerate(_pairs(d["curve"])):
            out.append(f"- frac {b/nb:.1f}-{(b+1)/nb:.1f}: within={_f(w)}  cross={_f(c)}")

    elif "position_curve" in k:                                     # 14
        out.append("**per-step-position (within | cross)**")
        for t, (w, c) in enumerate(_pairs(d["position_curve"])):
            out.append(f"- pos {t}: within={_f(w)}  cross={_f(c)}")
        if "window_results" in k:
            wr = d["window_results"].item()
            out.append("  windows: " + ", ".join(
                f"{n}={_f(v[0])}" for n, v in wr.items()))

    elif "w_raw" in k and "pr_neurons" in k:                        # 25
        out.append(f"**probe weight w interpretation** (band={d.get('band','?')})")
        out.append(f"- cos(w, mean-diff incorrect-correct) = {_f(d['cos_diffmeans'],3)}; "
                   f"cos(w, mean-act) = {_f(d['cos_meanact'],3)}; cos(w, sigma) = {_f(d['cos_sigma'],3)}")
        out.append(f"- sparsity: eff #neurons (PR) = {_f(d['pr_neurons'],1)}; "
                   f"50% mass in {int(d['n50'])} neurons, 90% in {int(d['n90'])}")
        tt = d.get('top_tokens', np.array([]))
        if len(tt):
            out.append(f"- logit-lens TOP: {list(tt[:15])}")
            out.append(f"- logit-lens BOT: {list(d.get('bot_tokens', np.array([]))[:15])}")

    elif "fail_base" in k and "fail_resid" in k:                    # 24
        out.append(f"**difficulty/failure decoupling (residualization)** (band={d.get('band','?')})")
        out.append(f"- (1) cos(w_diff,w_fail) = {_f(d['cos'],3)} (random ~ {_f(d.get('rand_cos',0),3)})")
        out.append(f"- (2) within-failure AUROC: base {_f(d['fail_base'],3)} -> "
                   f"after removing w_diff {_f(d['fail_resid'],3)}")
        out.append(f"- (3) difficulty corr^2: base {_f(d['diff_r2_base'],3)} -> "
                   f"after removing w_fail {_f(d['diff_r2_resid'],3)}")

    elif "methods" in k and "within" in k:                          # 23
        out.append(f"**trajectory amplifier vs simple pooling** (band={d.get('band','?')})")
        mn = d["methods"]; wi = d["within"]
        for j in range(len(mn)):
            tag = {"mean": "  <- simple", "attn": "  <- amplifier"}.get(str(mn[j]), "")
            out.append(f"- {str(mn[j]):5s} within={_f(wi[j])}{tag}")

    elif "bands" in k and "pooled" in k and "pairwise" in k:        # 22
        out.append("**pooled vs within-problem PAIRWISE training** "
                   f"(PCA={d.get('pca_k','?')})")
        bs = d["bands"]; po = d["pooled"]; pa = d["pairwise"]
        for j in range(len(bs)):
            out.append(f"- {str(bs[j]):6s} pooled={_f(po[j])}  pairwise={_f(pa[j])}"
                       f"  delta={_f(pa[j]-po[j])}")

    elif "feat_names" in k and "within" in k and "cloud_layer" in k:  # 21
        out.append(f"**token-cloud geometry vs pooled centroid** "
                   f"(layer={d.get('cloud_layer','?')}, k={d.get('k','?')}, "
                   f"tok_cap={d.get('tok_cap','?')})")
        fn = d["feat_names"]; wi = d["within"]
        for j in range(len(fn)):
            tag = "  <- pooled baseline" if str(fn[j]) == "centroid_spe" else ""
            out.append(f"- {str(fn[j]):13s} within={_f(wi[j])}{tag}")

    elif "comp_within" in k:                                        # 20
        out.append(f"**ensemble vs best single** (band={d.get('band','?')})")
        cn = d["comp_names"]; cw = d["comp_within"]
        for j in range(len(cn)):
            out.append(f"- {str(cn[j]):8s} within={_f(cw[j])}")
        out.append(f"- ensemble z-mean = {_f(d['ens_zmean'])} ; meta = {_f(d['ens_meta'])} "
                   f"; best single = {_f(d['best_single'])}")

    elif "ks" in k and "within" in k and "norm_within" in k:        # 19
        out.append(f"**manifold-constraint SPE: within-AUROC vs healthy-subspace dim k** "
                   f"(band={d.get('band','?')}, agg={d.get('agg','?')})")
        ks_ = d["ks"]; wi = d["within"]; cr = d.get("cross", np.full(len(wi), np.nan))
        for j in range(len(ks_)):
            out.append(f"- k={int(ks_[j]):<4d} within={_f(wi[j])}  cross={_f(cr[j])}")
        out.append(f"- ||z|| norm baseline (within) = {_f(d['norm_within'])}")

    elif "variants" in k and "within" in k:                         # 18
        out.append(f"**signal-strengthening ladder** (band={d.get('band','?')})")
        vs = d["variants"]; wi = d["within"]; ws = d.get("within_std", np.zeros(len(wi)))
        cr = d.get("cross", np.full(len(wi), np.nan))
        for j in range(len(vs)):
            out.append(f"- {str(vs[j]):38s} within={_f(wi[j])} +/- {_f(ws[j],3)}  cross={_f(cr[j])}")

    elif "l1" in k:                                                 # 17
        out.append("**sparse(L1) vs low-rank(PCA)**")
        l1 = d["l1"].item(); pca = d["pca"].item()
        out.append("- L1 (C: AUROC, #nonzero): " +
                   ", ".join(f"{c}:{_f(a)}/{int(nz) if nz==nz else 'nan'}"
                             for c, (a, nz) in l1.items()))
        out.append("- PCA (k: AUROC): " + ", ".join(f"{kk}:{_f(a)}" for kk, a in pca.items()))

    elif "results" in k:                                            # 13 / 11
        r = d["results"].item()
        out.append(f"**normalization / metric variants** (band={d.get('layer_band','?')})")
        for name2, v in r.items():
            try:
                out.append(f"- {name2}: {_f(v)}")
            except Exception:
                out.append(f"- {name2}: {v}")

    else:
        out.append(f"  keys: {sorted(k)}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/*.npz")
    ap.add_argument("--out", default="results_summary.md")
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    # skip raw extraction npz (huge) and the obsolete effective-rank / M_D outputs
    # (the old approach that printed all-nan here)
    skip = ("_sv.npz", "multisample_sv.npz", "unembedding", "healthy_baseline",
            "d_dynamics", "gsm8k_cim", "gsm8k_geom", "gsm8k_sv_analysis",
            "step_vector_analysis", "geometry_analysis",
            "_pr.npz", "_ae.npz")        # old per-step PR/AE per band (all ~chance, not informative)
    files = [f for f in files if not any(s in os.path.basename(f) for s in skip)]
    blocks = ["# Results summary (data/*.npz)\n"]
    for f in files:
        blocks.append(summarize(f))
    text = "\n".join(blocks)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\n[wrote {args.out} ; {len(files)} result files]")


if __name__ == "__main__":
    main()
