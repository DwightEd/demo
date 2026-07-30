from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_ghost_real_4090.sh"
CLI = PROJECT_ROOT / "src" / "crwh" / "ghost_cli.py"
EXTRACTOR = PROJECT_ROOT / "src" / "crwh" / "ghost_hf.py"


def test_ghost_script_is_a_distinct_fail_closed_real_data_pipeline() -> None:
    raw = SCRIPT.read_bytes()
    text = raw.decode("utf-8")

    assert raw.startswith(b"#!/usr/bin/env bash\n")
    assert b"\r\n" not in raw
    assert "set -Eeuo pipefail" in text
    assert "_FAILED" in text
    assert "_SUCCESS" in text
    assert (
        "/share/home/tm902089733300000/a903202310/lys/"
        "models/Meta-Llama-3.1-8B-Instruct"
    ) in text
    assert (
        "/share/home/tm902089733300000/a903202310/lys/"
        "research/demo/data/hf_datasets/ProcessBench"
    ) in text
    assert "nvidia-smi" in text
    assert "torch.cuda.is_available()" in text
    assert "torch.cuda.is_bf16_supported()" in text
    assert 'MAX_TOKENS="${MAX_TOKENS:-768}"' in text
    assert "-m crwh.ghost_cli select" in text
    assert "-m crwh.ghost_cli extract" in text
    assert "-m crwh.ghost_cli evaluate" in text
    assert "-m hypergraph.attention.cct" not in text
    assert "src/crwh/cohort.py" in text
    assert "hypergraph/attention/splitting.py" in text
    assert "hypergraph/attention/cct/processbench.py" in text
    assert "output_hidden_states=True" not in text
    assert "enable_grad" not in text


def test_remote_console_is_progress_only_while_details_stay_in_run_log() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")
    cli_text = CLI.read_text(encoding="utf-8")
    extractor_text = EXTRACTOR.read_text(encoding="utf-8")

    assert 'LOG_FILE="${RUN_DIR}/run.log"' in script_text
    assert 'exec 3>&1 4>&2' in script_text
    assert 'exec >>"${LOG_FILE}" 2>&1' in script_text
    assert 'tee -a "${LOG_FILE}" >&4' in script_text
    assert 'exec > >(tee -a "${RUN_DIR}/run.log") 2>&1' not in script_text
    assert 'printf "Completed: %s\\n" "${RUN_DIR}"' in script_text
    assert 'printf "Failed: %s (see %s)\\n"' in script_text
    assert 'print(json.dumps({' not in script_text
    assert "--progress" in script_text
    assert "print(json.dumps(manifest" not in cli_text
    assert "print(json.dumps(extraction_manifest" not in cli_text
    assert "print(json.dumps(summary" not in cli_text
    assert "from tqdm.auto import tqdm" in extractor_text
    assert 'desc="Extracting hidden states"' in extractor_text
    assert 'f"extracted {index}' not in extractor_text


def test_ghost_profiles_distinguish_smoke_pilot_and_full_science_runs() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    smoke = text.partition("  smoke)")[2].partition("    ;;")[0]
    pilot = text.partition("  pilot)")[2].partition("    ;;")[0]
    full = text.partition("  full)")[2].partition("    ;;")[0]

    assert 'LIMIT="${LIMIT:-48}"' in smoke
    assert 'BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-200}"' in smoke
    assert 'LIMIT="${LIMIT:-160}"' in pilot
    assert 'BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"' in pilot
    assert 'SELECTION_MODE="${SELECTION_MODE:-all_eligible}"' in full
    assert 'BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-2000}"' in full


def test_hf_path_uses_base_model_hooks_and_never_materializes_all_layers() -> None:
    cli_text = CLI.read_text(encoding="utf-8")
    extractor_text = EXTRACTOR.read_text(encoding="utf-8")

    assert "AutoModel.from_pretrained" in cli_text
    assert "AutoModelForCausalLM" not in cli_text
    assert "register_forward_hook" in extractor_text
    assert "torch.inference_mode()" in extractor_text
    assert "use_cache=False" in extractor_text
    assert "output_hidden_states=False" in extractor_text
    assert "output_hidden_states=True" not in extractor_text
    assert "selected.float().mean(dim=0)" in extractor_text
    assert "tokenize_chat_record" in extractor_text
