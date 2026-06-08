"""Re-organize all features under the ANSWER-based label (answer correct = correct,
ignore '####' format). The 26_comprehensive_stats.py already computed this as its
'lenient' policy; here we extract it cleanly per dataset into results_answer/.

Outputs (folder results_answer/):
  <dataset>_answer.json   per-feature: within & cross AUROC, cohen_d, descriptive
  comparison.json         feature x dataset table of within / cross
"""
import json, os, glob

SRC = "results"
DST = "results_answer"
os.makedirs(DST, exist_ok=True)

DATASETS = [("v1", "v1"), ("v2_custom", "v2_custom"), ("v2_5shot", "v2_5shot")]


def load(tag):
    for f in [f"{SRC}/{tag}.comprehensive_stats.json", f"{SRC}/{tag}_comprehensive_stats.json"]:
        if os.path.exists(f):
            return json.load(open(f, encoding="utf-8"))
    return None


def get(e):
    w = e.get("auroc_within_problem_paired_signed", {})
    c = e.get("auroc_cross_problem_signed", {})
    d = e.get("effect_size_signed_error_minus_correct", {})
    desc = e.get("descriptive", {})
    cor = desc.get("correct", {}); err = desc.get("error", {})
    return {
        "within": w.get("value"), "within_npairs": w.get("n_pairs"),
        "cross": c.get("value"), "cross_ci95": c.get("ci95"),
        "cohen_d": d.get("cohen_d"),
        "correct_mean": cor.get("mean"), "correct_std": cor.get("std"), "n_correct": cor.get("n"),
        "error_mean": err.get("mean"), "error_std": err.get("std"), "n_error": err.get("n"),
    }


comparison = {}
meta_all = {}
for tag, name in DATASETS:
    j = load(tag)
    if j is None:
        print("skip (not found):", tag); continue
    R = j["results"].get("lenient")
    if R is None:
        print("no lenient policy in", tag); continue
    sm = R.get("_section_meta", {})
    feats = {k: get(R[k]) for k in R if not k.startswith("_") and R[k].get("auroc_within_problem_paired_signed")}
    out = {"label_policy": "answer_correct_ignoring_format",
           "n_contrastive_problems": sm.get("n_contrastive_problems"),
           "meta": j.get("_meta", {}), "features": feats}
    json.dump(out, open(f"{DST}/{name}_answer.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    meta_all[name] = sm.get("n_contrastive_problems")
    for f, v in feats.items():
        comparison.setdefault(f, {})[name] = {"within": v["within"], "cross": v["cross"]}
    print(f"wrote {DST}/{name}_answer.json  ({len(feats)} features, contrastive={sm.get('n_contrastive_problems')})")

json.dump({"label_policy": "answer_correct_ignoring_format",
           "contrastive_problems": meta_all, "by_feature": comparison},
          open(f"{DST}/comparison.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"wrote {DST}/comparison.json")

# print a ranked table for the main dataset
main = json.load(open(f"{DST}/v2_5shot_answer.json", encoding="utf-8"))["features"]
rows = sorted(main.items(), key=lambda kv: -abs((kv[1]["within"] or 0.5) - 0.5))
print("\n=== v2_5shot answer-label, ranked by |within-0.5| ===")
print(f"{'feature':18s}{'within':>8s}{'cross':>8s}{'d':>7s}")
for f, v in rows:
    print(f"{f:18s}{v['within']:8.3f}{v['cross']:8.3f}{(v['cohen_d'] or 0):7.2f}")
