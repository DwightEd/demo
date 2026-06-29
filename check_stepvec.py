# check_stepvec.py — dump the full npz schema so the loader can be wired correctly
import argparse, numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True); keys = list(z.files)
    print("file:", args.npz)
    print("keys:", keys)
    if "stepvec" not in keys:
        print("KILL-PREREQ: no stepvec; re-extract with --store_step_vectors"); return
    SV = z["stepvec"]
    shp = next((np.asarray(v).shape for v in SV if v is not None and len(v)), None)
    nsv = shp[1] if shp and len(shp) >= 2 else None
    print("stepvec example (T,n_sv,d):", shp, "| n_sv =", nsv)
    print("sv_layers:", [int(x) for x in z["sv_layers"]] if "sv_layers" in keys else "ABSENT")
    print("layers_used:", [int(x) for x in z["layers_used"]] if "layers_used" in keys else "ABSENT")
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in keys else None
    print("cloud_feature_names:", cnames)
    print("geom_feature_names:", [str(x) for x in z["geom_feature_names"]] if "geom_feature_names" in keys else "ABSENT")
    if "stepcloud" in keys:
        sc0 = next((np.asarray(v).shape for v in z["stepcloud"] if v is not None and len(v)), None)
        print("stepcloud example (T,L,F):", sc0)
    print("'resultant' present:", bool(cnames and "resultant" in cnames))
    ges = z["gold_error_step"].astype(int)
    print("chains correct=%d err=%d | problems=%d" % (
        int((ges < 0).sum()), int((ges >= 0).sum()), len(np.unique(z["problem_ids"]))))


if __name__ == "__main__":
    main()
