#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  resolve_collect_request.sh --prompt "<request>" [--validate-only] [--print-plan]

Behavior:
  - Parses a natural-language request into a profile/config/iteration plan.
  - Validates referenced train config files exist.
  - Never submits sbatch.
EOF
}

PROMPT=""
VALIDATE_ONLY=0
PRINT_PLAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT="${2:-}"
      shift 2
      ;;
    --validate-only)
      VALIDATE_ONLY=1
      shift
      ;;
    --print-plan)
      PRINT_PLAN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$PROMPT" ]]; then
  echo "--prompt is required" >&2
  usage
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
TRAIN_CFG_DIR="${REPO_ROOT}/torchtitan/torchtitan/models"

lower_prompt="$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]')"

iterations=1
profile=""
single_config=""

if [[ "$lower_prompt" == *"30 iterations"* ]]; then
  iterations=30
fi
if [[ "$lower_prompt" == *"single iteration"* ]]; then
  iterations=1
fi

if [[ "$lower_prompt" == *"8 node"* && "$lower_prompt" == *"all configs"* ]]; then
  profile="all_configs_8node"
elif [[ "$lower_prompt" == *"all configs"* ]]; then
  profile="all_configs_4node"
fi

if [[ "$lower_prompt" == *"8b_tp_4"* ]]; then
  single_config="8b_tp_4"
  profile="single_config"
fi

if [[ -z "$profile" ]]; then
  echo "Could not resolve prompt to a known profile/config." >&2
  exit 3
fi

configs=()
models=()

if [[ "$profile" == "all_configs_4node" ]]; then
  configs=("1b_dp_4" "8b_tp_4" "16b_ep_4_tp_4" "70b_tp_8_dp_4")
  models=("llama3" "llama3" "deepseek_v3" "llama3")
elif [[ "$profile" == "all_configs_8node" ]]; then
  configs=("8b_tp_8_dp_4" "8b_tp_8_fsdp_4" "8b_tp_8_pp_4" "70b_tp_8_dp_4" "70b_tp_8_fsdp_4" "70b_tp_8_fsdp_2_pp_2")
  models=("llama3" "llama3" "llama3" "llama3" "llama3" "llama3")
else
  configs=("$single_config")
  models=("llama3")
fi

missing=0
for i in "${!configs[@]}"; do
  cfg="${configs[$i]}"
  model="${models[$i]}"
  cfg_path="${TRAIN_CFG_DIR}/${model}/train_configs/${model}_${cfg}.toml"
  if [[ ! -f "$cfg_path" ]]; then
    echo "MISSING: $cfg_path" >&2
    missing=1
  fi
done

if [[ "$PRINT_PLAN" -eq 1 || "$VALIDATE_ONLY" -eq 1 ]]; then
  echo "profile=${profile}"
  echo "iterations=${iterations}"
  echo "config_count=${#configs[@]}"
  for i in "${!configs[@]}"; do
    echo "config=${configs[$i]} model=${models[$i]}"
  done
fi

if [[ "$missing" -ne 0 ]]; then
  echo "Validation failed: one or more config files are missing." >&2
  exit 4
fi

if [[ "$VALIDATE_ONLY" -eq 1 ]]; then
  echo "Validation succeeded. No sbatch submission performed."
  exit 0
fi

if [[ "$PRINT_PLAN" -eq 1 ]]; then
  echo "Plan rendered. No sbatch submission performed."
  exit 0
fi

echo "No execution mode selected. Use --validate-only or --print-plan." >&2
exit 2
