#!/bin/bash
set -e

echo "============================================================"
echo "=== GRAPH ML EXPERIMENT BATCH - OGBN-ARXIV ================="
echo "============================================================"

# ============================================================
# DATA EXPLORATION
# ============================================================
echo ""
echo "============================================================"
echo "=== DATA EXPLORATION ======================================="
echo "============================================================"
python explore_data.py

# ============================================================
# 1. BASELINES
# ============================================================
echo ""
echo "============================================================"
echo "=== 1. BASELINES ==========================================="
echo "============================================================"

echo ""
echo "--- MLP: 2 layers ---"
python run_experiment.py --model mlp --number 1 --num_layers 2

echo ""
echo "--- MLP: 3 layers ---"
python run_experiment.py --model mlp --number 2 --num_layers 3

# ============================================================
# 2. GCN
# ============================================================
echo ""
echo "============================================================"
echo "=== 2. GCN ================================================="
echo "============================================================"

echo ""
echo "--- GCN: 2 layers ---"
python run_experiment.py --model gcn --number 1 --num_layers 2

echo ""
echo "--- GCN: 3 layers ---"
python run_experiment.py --model gcn --number 2 --num_layers 3

echo ""
echo "--- GCN: 2 layers, larger network ---"
# super long 
python run_experiment.py --model gcn --number 3 --num_layers 2 --hidden_channels 256

echo ""
echo "--- GCN: 3 layers, smaller network ---"
python run_experiment.py --model gcn --number 4 --num_layers 3 --hidden_channels 64

# ============================================================
# 3. RESIDUAL GCN
# ============================================================
echo ""
echo "============================================================"
echo "=== 3. RESIDUAL GCN ========================================"
echo "============================================================"

echo ""
echo "--- ResGCN: 2 layers ---"
python run_experiment.py --model resgcn --number 1 --num_layers 2 

# ============================================================
# 4. GRAPHSAGE
# ============================================================
echo ""
echo "============================================================"
echo "=== 4. GRAPHSAGE ==========================================="
echo "============================================================"

echo ""
echo "--- GraphSAGE: 2 layers ---"
python run_experiment.py --model sage --number 1 --num_layers 2

echo ""
echo "--- GraphSAGE: 3 layers ---"
python run_experiment.py --model sage --number 2 --num_layers 3

echo ""
echo "--- GraphSAGE: 2 layers, larger network ---"
python run_experiment.py --model sage --number 3 --num_layers 2 --hidden_channels 256

echo ""
echo "--- GraphSAGE: 3 layers, smaller network ---"
python run_experiment.py --model sage --number 4 --num_layers 3 --hidden_channels 64

# ============================================================
# 5. GAT
# ============================================================
echo ""
echo "============================================================"
echo "=== 5. GAT ================================================="
echo "============================================================"

echo ""
echo "--- GAT: Small to test running ---"
python run_experiment.py --model gat --number 0 --num_layers 2 --hidden_channels 64 --num_heads 2 --epochs 30

echo ""
echo "--- GAT: 2 layers ---"
echo "too long"
# python run_experiment.py --model gat --number 1 --num_layers 2

echo ""
echo "--- GAT: 3 layers ---"
echo "too long"
# python run_experiment.py --model gat --number 2 --num_layers 3

echo ""
echo "--- GAT: 2 layers, larger network ---"
echo "too long"
# python run_experiment.py --model gat --number 3 --num_layers 2 --hidden_channels 256

echo ""
echo "--- GAT: 3 layers, smaller network ---"
echo "not useful"
# python run_experiment.py --model gat --number 4 --num_layers 3 --hidden_channels 64

# ============================================================
# 6. ABLATIONS
# ============================================================
echo ""
echo "============================================================"
echo "=== 6. ABLATIONS ==========================================="
echo "============================================================"

echo ""
echo "--- Structure only: GraphSAGE + degree, 2 layers ---"
python run_experiment.py --model sage --number 1 --features degree --num_layers 2

echo ""
echo "--- Structure only: GraphSAGE + degree, 3 layers ---"
echo "too long"
# python run_experiment.py --model sage --number 2 --features degree --num_layers 3

echo ""
echo "--- Negative control: GCN + random features, 2 layers ---"
python run_experiment.py --model gcn --number 1 --features none --num_layers 2

echo ""
echo "--- Negative control: GCN + random features, 3 layers ---"
echo "too long"
# python run_experiment.py --model gcn --number 2 --features none --num_layers 3

# ============================================================
# FINAL COMPARISON
# ============================================================
echo ""
echo "============================================================"
echo "=== FINAL COMPARISON ======================================="
echo "============================================================"
python compare_results.py
