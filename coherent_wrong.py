"""Precondition for the mechanism-screening program (hypotheses.md H1-H3): how big is the
'coherent-but-wrong' subset = error steps with NORMAL/high kappa and LOW entropy (concentrated,
confident, but wrong)? If small, H1-H3 target only a sliver. Uses correct-step distributions as the
'normal' reference. Runs on _coh.npz (stepcloud resultant + tok_U_D entropy + gold_error_step)."""
from __future__ import annotations
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]; lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    fi = cn.index("resultant"); SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int); UD = z["tok_U_D"]
    KA_c, EN_c, KA_e, EN_e = [], [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i]); correct = k < 0; T = rng.shape[0]; a0 = int(rng[0, 0]); ud = np.asarray(UD[i], float)
        for j in range(T):
            if correct or j < k:
                tgt = (KA_c, EN_c)
            elif j == k:
                tgt = (KA_e, EN_e)
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(ud), int(rng[j, 1]) - a0 + 1)
            if hi - lo < 2 or not np.isfinite(sc[j, li, fi]):
                continue
            tgt[0].append(sc[j, li, fi]); tgt[1].append(float(np.nanmean(ud[lo:hi])))
    KA_c = np.asarray(KA_c); EN_c = np.asarray(EN_c); KA_e = np.asarray(KA_e); EN_e = np.asarray(EN_e)
    kmed = np.median(KA_c); emed = np.median(EN_c)   # 'normal' reference from correct steps
    hi_k = KA_e >= kmed                  # error step as concentrated as a typical correct step
    lo_e = EN_e <= emed                  # error step as confident as a typical correct step
    coh = hi_k & lo_e                    # coherent-but-wrong candidate
    print(f"{args.npz} | L{args.layer} | correct-steps {len(KA_c)} | error-steps {len(KA_e)}")
    print(f"  kappa: correct med {kmed:.3f}  error med {np.median(KA_e):.3f}")
    print(f"  entropy: correct med {emed:.3f}  error med {np.median(EN_e):.3f}")
    print(f"  error steps with HIGH kappa (>= correct median): {hi_k.mean()*100:5.1f}%")
    print(f"  error steps with LOW entropy (<= correct median): {lo_e.mean()*100:5.1f}%")
    print(f"  COHERENT-BUT-WRONG (high kappa AND low entropy):  {coh.mean()*100:5.1f}%  (n={int(coh.sum())})")


if __name__ == "__main__":
    main()
