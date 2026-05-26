"""Download ProcessBench and inspect its format."""
from datasets import load_dataset
import json
import os

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processbench")
os.makedirs(SAVE_DIR, exist_ok=True)

ds = load_dataset("Qwen/ProcessBench")
print("Splits:", list(ds.keys()))

for split in ds:
    print(f"\n=== {split}: {len(ds[split])} examples ===")
    print(f"Columns: {ds[split].column_names}")

    # Show first example
    ex = ds[split][0]
    for k, v in ex.items():
        val_str = str(v)
        if len(val_str) > 300:
            val_str = val_str[:300] + "..."
        print(f"  {k}: {val_str}")

    # Save to jsonl
    out_path = os.path.join(SAVE_DIR, f"{split}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(ds[split])):
            f.write(json.dumps(ds[split][i], ensure_ascii=False) + "\n")
    print(f"  Saved to {out_path}")

# Summary stats
print("\n=== Summary ===")
for split in ds:
    labels = [ex["label"] for ex in ds[split]]
    n_correct = sum(1 for l in labels if l == -1 or l is None)
    n_error = len(labels) - n_correct
    print(f"{split}: {len(labels)} total, {n_correct} correct (label=-1), {n_error} with errors")

    # Step count distribution
    step_counts = [len(ex["steps"]) for ex in ds[split]]
    print(f"  Steps: min={min(step_counts)}, max={max(step_counts)}, mean={sum(step_counts)/len(step_counts):.1f}")
