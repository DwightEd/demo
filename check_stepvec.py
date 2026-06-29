# check_stepvec.py — verify raw step vectors exist; report sv-layer mapping
import argparse, numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True); keys = set(z.files)
    print("file:", args.npz, "| has stepvec:", "stepvec" in keys)
    if "stepvec" not in keys:
        print("KILL-PREREQ: re-extract with --store_step_vectors --sv_layers 14"); return
    SV = z["stepvec"]; shp = next((np.asarray(v).shape for v in SV if v is not None and len(v)), None)
    print("n_chains:", len(SV), "| example (T,n_sv,d):", shp)
    print("sv_layers:", [int(x) for x in z["sv_layers"]] if "sv_layers" in keys else "None(n_sv==1)")
    print("cloud_feature_names:", [str(x) for x in z["cloud_feature_names"]])
    ges = z["gold_error_step"].astype(int)
    print("chains correct=%d err=%d | problems=%d" % (
        int((ges < 0).sum()), int((ges >= 0).sum()), len(np.unique(z["problem_ids"]))))


if __name__ == "__main__":
    main()
