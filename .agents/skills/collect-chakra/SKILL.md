---
name: collect-chakra
description: Collect ground truth Chakra traces
---

## Overview

This skill collects ground-truth Chakra traces from torchtitan by submitting Slurm jobs.

Invoke this skill when the user asks to collect Chakra traces and specifies any of:
- a specific config name (e.g. `8b_tp_4`)
- a profile scope (`all configs`, `4 node`, `8 node`)
- an iteration count

For static cluster policy (partitions, paths), refer to `.github/instructions/cluster-environments.instructions.md`.

---

## Step 1 — Resolve Intent

Map the user's prompt to a **profile** and **iteration count**. If you cannot map the user's prompt, ask the user for clarification.

| User prompt | Profile | Iterations |
|---|---|---|
| "Collect Chakra traces for all configs" | `all_configs_4node` | 1 (default) |
| "Collect Chakra traces for all configs for 30 iterations" | `all_configs_4node` | 30 |
| "Collect Chakra trace for 8b_tp_4 for a single iteration" | `single_config` (`8b_tp_4`) | 1 |
| "Collect Chakra traces for all configs for 8 node setup" | `all_configs_8node` | 1 (default) |

**Iteration loop bounds:** N iterations = loop `0..N-1`. Never use `0..N`.

---

## Step 2 — Look Up Configs

### Profile: `all_configs_4node`

| Config | Model |
|---|---|
| `1b_dp_4` | llama3 |
| `8b_tp_4` | llama3 |
| `16b_ep_4_tp_4` | deepseek_v3 |

### Profile: `all_configs_8node`

| Config | Model |
|---|---|
| `8b_tp_8_dp_4` | llama3 |
| `8b_tp_8_fsdp_4` | llama3 |
| `8b_tp_8_pp_4` | llama3 |
| `70b_tp_8_dp_4` | llama3 |
| `70b_tp_8_fsdp_4` | llama3 |
| `70b_tp_8_fsdp_2_pp_2` | llama3 |

---

## Step 3 — Set Environment Variables

All runs require these variables **that MUST be explicitly exported in the sbatch script**:

| Variable | Description | Critical? |
|---|---|---|
| `ROOT_PATH` | Cluster root path | Yes |
| `CLUSTER_NAME` | e.g. `vader`, `champollion` | Yes |
| `JOB_CONFIG_NAME` | **Config suffix ONLY (no model prefix)** e.g. `70b_fsdp_8` not `llama3_70b_fsdp_8` | **CRITICAL** |
| `MODEL_NAME` | `llama3` or `deepseek_v3` | Yes |
| `NNODES` | **Number of nodes** | **CRITICAL** |
| `NRANK_PER_NODE` | **Ranks per node** | **CRITICAL** |
| `NUM_RANKS` | Total ranks (computed) | Yes |
| `MODEL_FLAVOR` | Model flavor override | Yes (custom only) |
| `DIRPATH` | Output directory | Yes |
| `ITERATION` | Current iteration index | Yes |
| `RDZV_MASTER_HOSTNAME` | Rendezvous master host | Yes |
| `JOB_OUTPUT_DIR` | Per-iteration output root | Yes |

⚠️ **CRITICAL:** 
- `JOB_CONFIG_NAME` must be the config **suffix only** (e.g., `70b_fsdp_8`), not the full filename (e.g., `llama3_70b_fsdp_8`). The full path is constructed as `${MODEL_NAME}_${JOB_CONFIG_NAME}.toml`.
- If `NNODES` and `NRANK_PER_NODE` are not explicitly exported, `run_train_for_collect.sh` will silently use defaults (`NNODES=4`), causing incorrect cluster resource allocation.
- If `JOB_CONFIG_NAME` includes the model prefix, the config file lookup will fail (double prefix).

Model-specific overrides:
- llama3 configs: `MODEL_NAME=llama3`
- deepseek config: `MODEL_NAME=deepseek_v3`, `MODEL_FLAVOR=deepseek_v3`

All scripts run from `torchtitan/`.

Path rule for per-atomic outputs:
- Never write CSV/log outputs directly under `torchtitan/outputs/<cluster>/<config>`.
- Always write per-iteration under:
    - `torchtitan/outputs/<cluster>/<config>/<iteration>`
- Always write profile traces for that run under:
    - `torchtitan/outputs/<cluster>/<config>/<iteration>/profile_trace`

---

## Step 4 — Build the Sbatch Script

Start from the appropriate template:
- 4-node: `.agents/skills/collect-chakra/scripts/collect_profile_4node.template.sbatch`
- 8-node: `.agents/skills/collect-chakra/scripts/collect_profile_8node.template.sbatch`

**Slurm resource flags by profile:**
- 4-node: `--nodes=4 --ntasks=4 --ntasks-per-node=1`, `NRANK_PER_NODE=1`
- 8-node: `--nodes=4 --ntasks=32 --ntasks-per-node=8`, `NRANK_PER_NODE=8`

**Partition by cluster** (from cluster-environments instructions):
- Vader: `H200rhel94_lowpri`
- Champollion: `champollion` + `#SBATCH --comment='minicol;apm'`

Extend the template with a nested loop over configs and iterations that calls `single_config.sh` for each combination. `single_config.sh` calls `run_train_for_collect.sh` → `run_train.sh` → `combine_trace.sh`.

**Example (1 node, 8 GPUs per node, 1 iteration):**

```bash
export NNODES=1
export NRANK_PER_NODE=8
export NUM_RANKS=$((NRANK_PER_NODE * NNODES))
export JOB_CONFIG_NAME="70b_fsdp_8"
export MODEL_FLAVOR="llama3"

for iter in {0..0}; do
    export ITERATION=${iter}
    export JOB_OUTPUT_DIR="${SCRIPT_DIR}/outputs/${CLUSTER_NAME}/${JOB_CONFIG_NAME}/${ITERATION}"
    export DIRPATH="${JOB_OUTPUT_DIR}/profile_trace"
    export RDZV_MASTER_HOSTNAME="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
    bash single_config.sh
done
```

**Example (4 nodes, 8 GPUs per node, 30 iterations):**

```bash
export NNODES=4
export NRANK_PER_NODE=8
export NUM_RANKS=$((NRANK_PER_NODE * NNODES))

for JOB_CONFIG_NAME in "1b_dp_4" "8b_tp_4" "16b_ep_4_tp_4"; do
    export JOB_CONFIG_NAME
    for iter in {0..29}; do
        export ITERATION=${iter}
        export JOB_OUTPUT_DIR="${SCRIPT_DIR}/outputs/${CLUSTER_NAME}/${JOB_CONFIG_NAME}/${ITERATION}"
        export DIRPATH="${JOB_OUTPUT_DIR}/profile_trace"
        export RDZV_MASTER_HOSTNAME="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
        bash single_config.sh
    done
done
```

If the user specifically mentions to 'run training job only' or 'combine trace files only', use `run_train_for_collect.sh` or `combine_trace.sh` directly instead `single_config.sh` in the full loop.

---

## Step 4b — Script Management

**Script Location:** Save sbatch scripts to `torchtitan/outputs/sbatch/` with a descriptive name, e.g. `collect_1b_dp_4_30iter.sbatch`.

**Script Reuse:** Before creating a new sbatch script, check if a compatible one already exists in `torchtitan/outputs/sbatch/`. If a script matches the profile (configs, iteration count, cluster, node setup), reuse it instead of creating a duplicate.

**Job ID Tracking:** After a job is assigned a SLURM job ID, append a comment line to the sbatch script file:
```bash
# Used by job 34330
```

For multiple jobs using the same script:
```bash
# Used by job 34330, 34331, 34332
```

This helps audit which jobs have used which scripts.

---

## Step 5 — Validate Before Submitting

**MANDATORY VALIDATION CHECKS** before submitting any sbatch script:

1. **Use the validation script:**
   ```bash
   bash .agents/skills/collect-chakra/scripts/validate_sbatch.sh <sbatch_file>
   ```
   This script verifies:
   - `export NNODES=...` is present
   - `export NRANK_PER_NODE=...` is present
   - Both are in the loop body or before it
   - Values match SLURM resource allocation (`--nodes`, `--ntasks-per-node`)
   - `JOB_CONFIG_NAME` does NOT include model prefix (e.g., `70b_fsdp_8` ✓, not `llama3_70b_fsdp_8` ✗)

2. **Check script and config files:**
   - All referenced scripts exist (`single_config.sh`, `run_train_for_collect.sh`, `combine_trace.sh`)
   - All config files exist: `torchtitan/models/${MODEL_NAME}/train_configs/${MODEL_NAME}_${JOB_CONFIG_NAME}.toml`
   - The venv can be activated
   - The sbatch script is saved to `torchtitan/outputs/sbatch/`

3. **Verify bash syntax:**
   ```bash
   bash -n <sbatch_file>
   ```

---

## Step 6 — Submit and Track

After `sbatch` returns a job ID, append it to the sbatch script file in `torchtitan/outputs/sbatch/` using the following script:

```bash
#!/bin/bash
SBATCH_FILE="torchtitan/outputs/sbatch/<script_name>.sbatch"
JOB_ID="<JOBID>"

if tail -1 "$SBATCH_FILE" | grep -q "# Used by job"; then
    # Append to existing line
    sed -i "$ s/$/, $JOB_ID/" "$SBATCH_FILE"
else
    # Create new line
    echo "# Used by job $JOB_ID" >> "$SBATCH_FILE"
fi
```

---

## Expected Outputs

Per-iteration files under `JOB_OUTPUT_DIR/`, which is resolved in run_train_for_collect.sh:
- `collect_<config>_iter<iteration>.log`
- `profile_trace/comm_groups*.json`
- `profile_trace/host_device_trace*.json`
- `profile_trace/kineto_rank*.json`
- `profile_trace/pytorch_et_rank*.json`
- `profile_trace/trace*.et`
- `profile_trace/trace*.json`
- `bw_watch.csv`

Per-job files under `outputs/<slurm_job_id>/`:
- `sbatch.out`
- Champollion only: `minicol.jobid-*`, `apm.jobid-*` (move into job output directory)

---

## Guardrails

- Do not overwrite existing per-iteration output directories.
- Do not submit an `sbatch` job for validation-only requests.
- **ALWAYS export NNODES and NRANK_PER_NODE explicitly in the sbatch script.** Silent defaults in `run_train_for_collect.sh` cause misconfigurations.
- **ALWAYS set JOB_CONFIG_NAME WITHOUT the model prefix.** Use `70b_fsdp_8`, not `llama3_70b_fsdp_8`. CONFIG_FILE is built as `${MODEL_NAME}_${JOB_CONFIG_NAME}.toml`.
- **ALWAYS run validate_sbatch.sh before submitting.** This catches missing exports, format errors, and misaligned SLURM settings.
