# CTG Data Paths

Do **not** use:

```bash
data/full_gsm8k.npz
```

That path is wrong for the current full-trace files.

Use the canonical remote paths:

```bash
NPZ=/gz-data/research/demo/data/features/full_gsm8k.npz
HIDDEN=/gz-data/research/demo/data/hidden/gsm8k
```

From `/gz-data/research/demo`, the relative paths are:

```bash
NPZ=data/features/full_gsm8k.npz
HIDDEN=data/hidden/gsm8k
```

Run CTG:

```bash
cd /gz-data/research/demo
python constraint_transport_geometry_audit.py \
  data/features/full_gsm8k.npz \
  --hidden_dir data/hidden/gsm8k \
  --layers 10,14,18,22 \
  --device cuda \
  --bootstrap 300
```

Or set environment variables once:

```bash
export CTG_NPZ=/gz-data/research/demo/data/features/full_gsm8k.npz
export CTG_HIDDEN_DIR=/gz-data/research/demo/data/hidden/gsm8k

python constraint_transport_geometry_audit.py \
  --layers 10,14,18,22 \
  --device cuda \
  --bootstrap 300
```

If the server layout changed, locate the files with:

```bash
find /gz-data/research/demo/data -name 'full_gsm8k.npz' -o -name '*gsm8k*.npz'
find /gz-data/research/demo/data -type d -path '*hidden*gsm8k*'
```

Known full-trace convention:

```text
data/features/full_<subset>.npz
data/hidden/<subset>/<id>.npy
```

For example:

```text
data/features/full_gsm8k.npz
data/hidden/gsm8k/gsm8k-0.npy
```
