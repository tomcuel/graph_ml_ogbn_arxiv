"""
Example: reload a trained model from checkpoints/ and re-run evaluation

Usage:
    python load_checkpoint_example.py --checkpoint checkpoints/gat_full.pt
"""

import argparse

import torch

from src.data import load_arxiv
from src.models import build_model
from src.train import evaluate, load_checkpoint
from ogb.nodeproppred import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    config = ckpt["config"]

    print(f"Using device: {device}")
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Config: {config}")
    print(f"Saved scores -> val: {ckpt['val_acc']:.4f} | test: {ckpt['test_acc']:.4f}")

    if config["model"] == "gat":
        model = build_model(config["model"], config["in_channels"], config["hidden_channels"], config["out_channels"], config["num_layers"], config["dropout"], num_heads=config["num_heads"]).to(device)
    else:
        model = build_model(config["model"], config["in_channels"], config["hidden_channels"], config["out_channels"], config["num_layers"], config["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Re-run evaluation from scratch to confirm the checkpoint reproduces the same scores
    # (only meaningful for features="full"; degree/none features would need to be
    # regenerated identically to match training-time features).
    data, split_idx, _ = load_arxiv(root=args.data_root)
    data = data.to(device)
    split_idx = {k: v.to(device) for k, v in split_idx.items()}

    evaluator = Evaluator(name="ogbn-arxiv")
    results = evaluate(model, data, split_idx, evaluator)
    print(f"Re-evaluated scores -> train: {results['train']:.4f} | val: {results['valid']:.4f} | test: {results['test']:.4f}")


if __name__ == "__main__":
    main()
