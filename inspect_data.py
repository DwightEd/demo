# inspect_data.py — confirm the actual contents of a feature npz, or a hidden-shard dir.
# Usage:
#   python inspect_data.py data/features/full_gsm8k.npz
#   python inspect_data.py data/hidden/gsm8k          # samples a few <id>.npy shards
#   python inspect_data.py data/gsm8k_v2_5shot.npz
import argparse, os, numpy as np


def _ex(z, name):
    v = z[name]
    if getattr(v, "dtype", None) == object:
        shp = next((np.asarray(x).shape for x in v if x is not None and np.asarray(x).size), None)
        return f"object[{len(v)}]  eg-shape {shp}"
    return f"{v.dtype} {v.shape}"


def show_npz(path):
    z = np.load(path, allow_pickle=True); keys = list(z.files)
    print(f"== NPZ {path} ==")
    print("keys:", keys)
    for k in keys:
        print(f"  {k}: {_ex(z, k)}")
    if "gold_error_step" in keys:
        g = z["gold_error_step"].astype(int)
        print(f"  -> chains={len(g)}  correct={int((g < 0).sum())}  err={int((g >= 0).sum())}")
    if "problem_ids" in keys:
        print(f"  -> unique problems={len(np.unique(z['problem_ids']))}")
    for meta in ("cloud_feature_names", "layers_used", "sv_layers", "hidden_layers"):
        if meta in keys:
            vals = z[meta]
            try:
                shown = [str(x) for x in vals] if meta == "cloud_feature_names" else [int(x) for x in vals]
            except Exception:
                shown = list(vals)
            print(f"  -> {meta} = {shown}")
    if "hidden_stored" in keys and bool(z["hidden_stored"]):
        hd = str(z["hidden_dir"]); hl = [int(x) for x in z["hidden_layers"]] if "hidden_layers" in keys else "?"
        print(f"  -> HIDDEN STORED: dir={hd}  layers={hl}")
        hf = list(z["hidden_files"]) if "hidden_files" in keys else []
        if hf:
            cand = str(hf[0]) if os.path.isabs(str(hf[0])) else os.path.join(hd, str(hf[0]))
            ok = os.path.exists(cand)
            print(f"     shard0={hf[0]}  resolves={ok}", end="")
            if ok:
                a = np.load(cand, mmap_mode="r"); print(f"  shape {a.shape}  dtype {a.dtype}", end="")
            print()


def show_dir(path):
    npys = sorted(f for f in os.listdir(path) if f.endswith(".npy"))
    print(f"== DIR {path}/  ({len(npys)} .npy shards) ==")
    for f in npys[:3]:
        a = np.load(os.path.join(path, f), mmap_mode="r")
        print(f"  {f}: shape {a.shape}  dtype {a.dtype}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); a = ap.parse_args()
    (show_dir if os.path.isdir(a.path) else show_npz)(a.path)


if __name__ == "__main__":
    main()
