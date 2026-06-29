# nts/data/loader.py — load an extracted ProcessBench npz into StepTable (robust to schema variance)
import re
import numpy as np
from ..core.types import ChainData, StepTable


def _stepvec_layers(z, n_sv):
    """Which model layers the stepvec stores. Prefer explicit sv_layers; else assume == layers_used."""
    if "sv_layers" in z.files:
        return [int(x) for x in z["sv_layers"]]
    if "layers_used" in z.files:
        lyu = [int(x) for x in z["layers_used"]]
        if len(lyu) == n_sv:
            return lyu
    return list(range(n_sv))   # unknown mapping -> positional indices


def _nearest(layers, layer):
    return int(np.argmin(np.abs(np.asarray(layers) - layer)))


def _rep_rate(text):
    toks = re.findall(r"\w+", str(text).lower())
    if len(toks) < 6:
        return 0.0
    tri = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    return 1.0 - len(set(tri)) / len(tri)


def load_step_table(npz, layer=14, verbose=True):
    z = np.load(npz, allow_pickle=True)
    if "stepvec" not in z.files:
        raise RuntimeError(f"{npz} has no stepvec; re-extract with --store_step_vectors")
    SV = z["stepvec"]
    nsv = next((np.asarray(v).shape[1] for v in SV if v is not None and len(v)), 1)
    sv_layers = _stepvec_layers(z, nsv)
    svi = _nearest(sv_layers, layer); sv_used = sv_layers[svi]
    if verbose and sv_used != layer:
        print(f"[loader] requested layer {layer} not in stepvec layers {sv_layers}; using nearest = {sv_used} (idx {svi})")

    ges = z["gold_error_step"].astype(int); pid = z["problem_ids"].astype(int)
    ranges = z["step_token_ranges"]; texts = z["steps_text"]

    # kappa (resultant) is OPTIONAL — only gate2/gate3's coherent-but-wrong region uses it
    kappa_ok = False; ri = li = None; SC = None
    if "stepcloud" in z.files and "cloud_feature_names" in z.files:
        cnames = [str(x) for x in z["cloud_feature_names"]]
        if "resultant" in cnames:
            ri = cnames.index("resultant")
            lyu = [int(x) for x in z["layers_used"]] if "layers_used" in z.files else list(range(99))
            li = _nearest(lyu, layer); SC = z["stepcloud"]; kappa_ok = True
    if verbose and not kappa_ok:
        avail = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
        print(f"[loader] 'resultant'(kappa) not in cloud features {avail}; kappa=NaN -> gate2/3 cbw region auto-skipped")

    chains = []
    for i in range(len(SV)):
        v = SV[i]
        if v is None or len(v) == 0:
            continue
        vecs = np.asarray(v)[:, svi, :].astype(np.float32); T = len(vecs)
        rr = np.asarray(ranges[i]); txt = texts[i]
        y = np.array([1 if (ges[i] >= 0 and t == ges[i]) else 0 for t in range(T)])
        length = (rr[:, 1] - rr[:, 0]).astype(float)
        speed = np.full(T, np.nan)
        for t in range(1, T):
            speed[t] = np.linalg.norm(vecs[t] - vecs[t - 1])
        rep = np.array([_rep_rate(txt[t]) if t < len(txt) else 0.0 for t in range(T)])
        if kappa_ok:
            sc = np.asarray(SC[i]); kappa = np.array([float(sc[t, li, ri]) for t in range(T)])
        else:
            kappa = np.full(T, np.nan)
        chains.append(ChainData(vecs=vecs, y=y, length=length, speed=speed,
                                repetition=rep, kappa=kappa, problem_id=int(pid[i]), correct=ges[i] < 0))
    return StepTable(chains=chains)


def load_layer_matrix(npz, sv_index):
    """All correct-chain step vectors stacked at stored sv index (for ID curve)."""
    z = np.load(npz, allow_pickle=True); SV = z["stepvec"]; ges = z["gold_error_step"].astype(int)
    return np.concatenate([np.asarray(SV[i])[:, sv_index, :].astype(np.float32)
                           for i in range(len(SV)) if ges[i] < 0 and SV[i] is not None and len(SV[i])], 0)
