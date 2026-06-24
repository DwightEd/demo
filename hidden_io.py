"""Stream per-chain full hidden states dumped by extract_features --hidden_dump_dir."""
import os
import numpy as np


def _fn(cid):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(cid)) + ".npy"


def load_chain(hidden_dir, chain_id, mmap=True):
    """(R, L, d) fp16 full per-token hidden for one chain; mmap so only touched pages load."""
    return np.load(os.path.join(hidden_dir, _fn(chain_id)), mmap_mode="r" if mmap else None)


def layer_col(z, layer):
    """Column of a hidden_states layer number in the dumped (R, L, d) array."""
    return [int(x) for x in z["hidden_layers"]].index(layer)
