"""
Train a model on OGBN-Arxiv, log the result to results/results.csv, and save the trained weights to checkpoints/

Examples:
    # Experiment 1 - baseline (content only, no graph)
    python run_experiment.py --model mlp

    # Experiment 2 - GCN (content + graph)
    python run_experiment.py --model gcn
    
    # Experiment 3 - ResGCN (content + graph)
    python run_experiment.py --model resgcn

    # Experiment 4 - GraphSAGE
    python run_experiment.py --model sage

    # Experiment 5 - GAT
    python run_experiment.py --model gat

    # Ablation: "does structure alone help?"
    # Content features are replaced by the node's degree (one-hot), so the model only has access to structural information.
    python run_experiment.py --model sage --features degree

    # Ablation: random features (negative control, neither content nor structure carries real signal)
    python run_experiment.py --model gcn --features none
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

from src.data import load_arxiv
from src.models import build_model
from src.train import run_training, save_checkpoint


def apply_feature_ablation(data, mode, seed):
    """
    Apply feature ablation to the data
        mode='full'   -> original content features (title/abstract embedding)
        mode='degree' -> purely structural features (one-hot degree), no content
        mode='none'   -> random noise, neither content nor structure (negative control)
    """
    if mode == "full":
        return data
    if mode == "degree":
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes).long().clamp(max=255)
        data.x = F.one_hot(deg, num_classes=256).float()
    elif mode == "none":
        g = torch.Generator().manual_seed(seed)
        data.x = torch.randn(data.num_nodes, data.x.size(-1), generator=g)
    else:
        raise ValueError(f"Unknown feature mode: {mode}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "gcn", "resgcn", "sage", "gat"], help="model architecture")
    parser.add_argument("--number", type=int, default=1, help="run number (for logging multiple runs)")
    parser.add_argument("--features", type=str, default="full", choices=["full", "degree", "none"], help="full = real content, degree = structure only, none = negative control")
    parser.add_argument("--hidden_channels", type=int, default=128, help="hidden layer size")
    parser.add_argument("--num_layers", type=int, default=3, help="number of layers (including input and output)")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--num_heads", type=int, default=None, help="number of attention heads (for GAT only)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay (L2 regularization)")
    parser.add_argument("--epochs", type=int, default=100, help="maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience (in epochs)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--data_root", type=str, default="./data", help="root folder for OGBN-Arxiv dataset")
    parser.add_argument("--results_file", type=str, default="results/results.csv", help="CSV file to log results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="folder where the best model weights are saved")
    parser.add_argument("--num_pairs", type=int, default=10000, help="number of node pairs for over-smoothing diagnostic")
    parser.add_argument("--max_distance", type=int, default=5, help="maximum distance for over-squashing diagnostic")
    parser.add_argument("--n_targets", type=int, default=20, help="number of target nodes for over-squashing diagnostic")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data, split_idx, num_classes = load_arxiv(root=args.data_root)
    data = apply_feature_ablation(data, args.features, args.seed)
    data = data.to(device)
    split_idx = {k: v.to(device) for k, v in split_idx.items()}

    model = build_model(args.model, data.x.size(-1), args.hidden_channels, num_classes, args.num_layers, args.dropout, num_heads=args.num_heads).to(device)
    print(f"\n=== {args.model.upper()} | features={args.features} | {sum(p.numel() for p in model.parameters()):,} parameters ===\n")

    model, best_val, best_test, history, diagnostics = run_training(model, data, split_idx, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience, verbose=True, num_pairs=args.num_pairs, max_distance=args.max_distance, n_targets=args.n_targets)
    print(f"\n>>> {args.model.upper()} ({args.features}) — val acc: {best_val:.4f} | test acc: {best_test:.4f}")

    run_name = f"{args.model}_{args.features}_{args.number}"
    print(f"Run name: {run_name}")

    # --- log results to CSV ---
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    write_header = not os.path.exists(args.results_file)
    with open(args.results_file, "a") as f:
        if write_header:
            f.write("model,features,number,hidden_channels,num_layers,dropout,lr,val_acc,test_acc,checkpoint\n")
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{run_name}.pt")
        f.write(f"{args.model},{args.features},{args.number},{args.hidden_channels},{args.num_layers}, {args.dropout},{args.lr},{best_val:.4f},{best_test:.4f},{checkpoint_path}\n")

    # --- save training history (loss/accuracy per epoch) ---
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/history", exist_ok=True)
    hist_path = f"results/history/history_{run_name}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f)
    print(f"History saved: {hist_path}")

    # -- - save post-training diagnostics (oversmoothing metrics) ---
    if diagnostics is not None:
        os.makedirs("results", exist_ok=True)
        os.makedirs("results/diagnostics", exist_ok=True)
        diag_path = f"results/diagnostics/diagnostics_{run_name}.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f)
        print(f"Diagnostics saved: {diag_path}")

    # --- save the trained model itself ---
    config = {
        "model": args.model,
        "features": args.features,
        "in_channels": data.x.size(-1),
        "hidden_channels": args.hidden_channels,
        "out_channels": num_classes,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    if args.model == "gat":
        config["num_heads"] = args.num_heads
    save_checkpoint(model, checkpoint_path, config=config, val_acc=best_val, test_acc=best_test)
    print(f"Model checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
