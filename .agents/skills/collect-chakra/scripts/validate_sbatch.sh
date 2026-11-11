#!/bin/bash
# Validation script for sbatch files used in collect-chakra
# Checks for common configuration errors before submission

set -u

SBATCH_FILE="${1:?Usage: validate_sbatch.sh <sbatch_file>}"

if [[ ! -f "$SBATCH_FILE" ]]; then
    echo "❌ ERROR: Sbatch file not found: $SBATCH_FILE"
    exit 1
fi

ERRORS=0
WARNINGS=0

echo "🔍 Validating: $SBATCH_FILE"
echo ""

# Check 1: NNODES export
if grep -q "export NNODES=" "$SBATCH_FILE"; then
    NNODES_VALUE=$(grep "export NNODES=" "$SBATCH_FILE" | head -1 | sed 's/.*NNODES=\([0-9]*\).*/\1/')
    echo "✓ NNODES export found: $NNODES_VALUE"
    
    # Check if NNODES matches SLURM --nodes
    SLURM_NODES=$(grep "^#SBATCH --nodes=" "$SBATCH_FILE" | sed 's/.*--nodes=\([0-9]*\).*/\1/')
    if [[ -n "$SLURM_NODES" ]] && [[ "$NNODES_VALUE" != "$SLURM_NODES" ]]; then
        echo "⚠️  WARNING: NNODES ($NNODES_VALUE) does not match SLURM --nodes ($SLURM_NODES)"
        ((WARNINGS++))
    fi
else
    echo "❌ ERROR: export NNODES= not found in sbatch script"
    ((ERRORS++))
fi

echo ""

# Check 2: NRANK_PER_NODE export
if grep -q "export NRANK_PER_NODE=" "$SBATCH_FILE"; then
    NRANK_VALUE=$(grep "export NRANK_PER_NODE=" "$SBATCH_FILE" | head -1 | sed 's/.*NRANK_PER_NODE=\([0-9]*\).*/\1/')
    echo "✓ NRANK_PER_NODE export found: $NRANK_VALUE"
    
    # Check if NRANK_PER_NODE matches SLURM --ntasks-per-node
    SLURM_NTASKS_PER_NODE=$(grep "^#SBATCH --ntasks-per-node=" "$SBATCH_FILE" | sed 's/.*--ntasks-per-node=\([0-9]*\).*/\1/')
    if [[ -n "$SLURM_NTASKS_PER_NODE" ]] && [[ "$NRANK_VALUE" != "$SLURM_NTASKS_PER_NODE" ]]; then
        echo "⚠️  WARNING: NRANK_PER_NODE ($NRANK_VALUE) does not match SLURM --ntasks-per-node ($SLURM_NTASKS_PER_NODE)"
        ((WARNINGS++))
    fi
else
    echo "❌ ERROR: export NRANK_PER_NODE= not found in sbatch script"
    ((ERRORS++))
fi

echo ""

# Check 3: JOB_CONFIG_NAME format (should NOT include model prefix)
# Check both direct assignments and export statements
if grep -q "JOB_CONFIG_NAME=" "$SBATCH_FILE"; then
    JOB_CONFIG=$(grep 'JOB_CONFIG_NAME="' "$SBATCH_FILE" | head -1 | sed 's/.*JOB_CONFIG_NAME="\([^"]*\)".*/\1/')
    
    # Check if JOB_CONFIG_NAME includes model prefix (llama3_, deepseek_v3_)
    if [[ "$JOB_CONFIG" =~ ^llama3_ ]] || [[ "$JOB_CONFIG" =~ ^deepseek_v3_ ]]; then
        echo "❌ ERROR: JOB_CONFIG_NAME includes model prefix: '$JOB_CONFIG'"
        echo "   Should be: '${JOB_CONFIG#*_}' (without model prefix)"
        echo "   CONFIG_FILE is built as: \${MODEL_NAME}_\${JOB_CONFIG_NAME}.toml"
        ((ERRORS++))
    else
        echo "✓ JOB_CONFIG_NAME format correct: $JOB_CONFIG"
    fi
else
    echo "⚠️  WARNING: JOB_CONFIG_NAME= not found (may be dynamic)"
fi

echo ""

# Check 4: MODEL_NAME export
if grep -q "export MODEL_NAME=" "$SBATCH_FILE"; then
    MODEL_NAME=$(grep "export MODEL_NAME=" "$SBATCH_FILE" | head -1 | sed 's/.*MODEL_NAME="\([^"]*\)".*/\1/')
    echo "✓ MODEL_NAME export found: $MODEL_NAME"
else
    echo "⚠️  WARNING: export MODEL_NAME= not found"
fi

echo ""

# Check 5: NUM_RANKS calculation
if grep -q "export NUM_RANKS=" "$SBATCH_FILE"; then
    echo "✓ NUM_RANKS export found"
else
    echo "⚠️  WARNING: export NUM_RANKS= not found"
fi

echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ $ERRORS -eq 0 ]]; then
    echo "✓ Validation passed!"
else
    echo "❌ Validation failed with $ERRORS error(s)"
fi

if [[ $WARNINGS -gt 0 ]]; then
    echo "⚠️  $WARNINGS warning(s) found"
fi

exit $ERRORS
