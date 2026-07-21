#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="${REPO_ROOT_OVERRIDE:-${DEFAULT_REPO_ROOT}}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -d "${REPO_ROOT}" ]] || die "repository root does not exist: ${REPO_ROOT}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"

validate_directory_pair() {
  local source="$1"
  local destination="$2"

  if [[ ! -e "${source}" ]]; then
    return
  fi
  [[ -d "${source}" ]] || die "source is not a directory: ${source}"
  [[ ! -e "${destination}" ]] || \
    die "destination already exists; refusing to merge automatically: ${destination}"
}

migrate_directory() {
  local source="$1"
  local destination="$2"

  if [[ ! -e "${source}" ]]; then
    printf 'skip missing: %s\n' "${source}"
    return
  fi

  mkdir -p "$(dirname "${destination}")"
  mv -- "${source}" "${destination}"
  printf 'moved: %s -> %s\n' "${source}" "${destination}"
}

cd "${REPO_ROOT}"

SOURCES=(
  "${REPO_ROOT}/outputs/attention_cohorts"
  "${REPO_ROOT}/outputs/attention_traces"
  "${REPO_ROOT}/outputs/attention_hypergraph"
)
DESTINATIONS=(
  "${REPO_ROOT}/data/attention_cohorts"
  "${REPO_ROOT}/data/attention_traces"
  "${REPO_ROOT}/results/attention_hypergraph"
)

for index in "${!SOURCES[@]}"; do
  validate_directory_pair "${SOURCES[$index]}" "${DESTINATIONS[$index]}"
done
for index in "${!SOURCES[@]}"; do
  migrate_directory "${SOURCES[$index]}" "${DESTINATIONS[$index]}"
done

printf '\nArtifact layout migration complete.\n'
printf 'Intermediate data: %s\n' "${REPO_ROOT}/data"
printf 'Experiment results: %s\n' "${REPO_ROOT}/results"
