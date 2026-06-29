# scripts/run_gate.py — Hydra entry: compose cfg -> load table -> run gate -> print/save
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make `nts` importable
import hydra
from omegaconf import DictConfig, OmegaConf
from nts.core.config import GeomCfg
from nts.core.registry import GATES
from nts.data.loader import load_step_table
import nts.signals  # register signals
import nts.gates    # register gates


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    root = hydra.utils.get_original_cwd()
    npz = os.path.join(root, cfg.data_dir, cfg.data.file)
    geom = GeomCfg(**OmegaConf.to_container(cfg.geom, resolve=True))
    table = load_step_table(npz, geom.layer)
    params = {"npz": npz}  # gate1 needs npz for the ID curve; others ignore it
    res = GATES.create(cfg.gate.name, cfg=geom, params=params).run(table)
    print(f"\n=== {cfg.gate.name} on {cfg.data.name} (layer {geom.layer}) ===")
    print(res.summary)
    out = os.path.join(root, cfg.outputs_dir); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{cfg.gate.name}_{cfg.data.name}_L{geom.layer}.json"), "w", encoding="utf-8") as fh:
        json.dump({"gate": cfg.gate.name, "data": cfg.data.name, "layer": geom.layer,
                   "kill": res.kill, "lines": res.lines}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
