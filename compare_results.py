"""
Aggregate results/results.csv, print a comparison table, and generate
several figures:
  - comparison_models.png     : MLP vs GCN vs GraphSAGE vs GAT (test accuracy)
  - comparison_ablation.png   : content-only vs structure-only vs content+structure
  - training_curves.png       : validation accuracy over epochs, one line per model
  - overfitting_gap.png       : train accuracy - val accuracy per model (overfitting check)

Also exports results/summary.csv and results/summary.md.

Usage:
    python compare_results.py
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/figures/comparison", exist_ok=True)
os.makedirs("results/history", exist_ok=True)
os.makedirs("results/diagnostics", exist_ok=True)

RESULTS_FILE = "results/results.csv"
FIG_DIR = "results/figures/comparison"
PALETTE = {"mlp": "#8172B2", "gcn": "#4C72B0", "sage": "#55A868", "gat": "#DD8452"}


def load_results(results_file=RESULTS_FILE):
    if not os.path.exists(results_file):
        print(f"No results found at {results_file}, run a few `python run_experiment.py --model ...` first")
        return None
    return pd.read_csv(results_file)


def export_results_md(df, out_dir_results="results"):
    md_path = os.path.join(out_dir_results, "results.md")
    with open(md_path, "w") as f:
        f.write("# Results\n\n")
        f.write(df[["model", "number", "features", "hidden_channels", "num_layers", "dropout", "lr", "val_acc", "test_acc"]].to_markdown(index=False))
        f.write("\n")
    return md_path


def plot_model_comparison(df, out_dir):
    """
    MLP vs GCN vs GraphSAGE vs GAT, all using full content+graph features
    """
    full = df[df["features"] == "full"].sort_values("test_acc")
    if full.empty:
        return None

    colors = [PALETTE.get(m, "#4C72B0") for m in full["model"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(full["model"] + " (" + full["number"].astype(str) + ")", full["test_acc"], color=colors)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Model comparison — content + graph (full features)")
    ax.set_ylim(0, min(1.0, full["test_acc"].max() + 0.15))
    for bar, v in zip(bars, full["test_acc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_models.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ablation(df, out_dir):
    """
    Content-only vs structure-only vs content+structure
    """
    rows = []
    mlp_full = df[(df["model"] == "mlp") & (df["features"] == "full")]
    if not mlp_full.empty:
        rows.append((f"Content only\n(MLP ({mlp_full['number'].values[0]}))", mlp_full["test_acc"].values[0], "#8172B2"))

    graph_models = df[df["model"] != "mlp"]
    struct_only = graph_models[graph_models["features"] == "degree"]
    if not struct_only.empty:
        r = struct_only.loc[struct_only["test_acc"].idxmax()]
        rows.append((f"Structure only\n({r['model']} ({r['number']}))", r["test_acc"], "#DD8452"))

    content_and_graph = graph_models[graph_models["features"] == "full"]
    if not content_and_graph.empty:
        r = content_and_graph.loc[content_and_graph["test_acc"].idxmax()]
        rows.append((f"Content + structure\n({r['model']} ({r['number']}))", r["test_acc"], "#55A868"))

    if len(rows) < 2:
        return None

    labels, values, colors = zip(*rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Which matters more: content or citation structure?")
    ax.set_ylim(0, min(1.0, max(values) + 0.15))
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_ablation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_histories(features="full"):
    """
    Load results/history_<run_name>.json for all runs where run_name = <model>_<features>_<number>
    """
    histories = {}
    for path in glob.glob(f"results/history/history_*_{features}_*.json"):
        model_name = os.path.basename(path).replace("history_", "").replace(f"_{features}.json", "")
        with open(path) as f:
            histories[model_name] = json.load(f)
    return histories


def plot_training_curves(out_dir, features="full"):
    """
    Validation accuracy across epochs, one line per model
    """
    histories = load_histories(features)
    if not histories:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model_name, history in histories.items():
        epochs = [h["epoch"] for h in history]
        val_acc = [h["valid"] for h in history]
        ax.plot(epochs, val_acc, label=model_name, color=PALETTE.get(model_name), linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(f"Training curves (validation accuracy, features={features})")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_overfitting_gap(out_dir, features="full"):
    """
    Final train_acc - val_acc per model: a quick overfitting sanity check
    """
    histories = load_histories(features)
    if not histories:
        return None

    models, gaps = [], []
    for model_name, history in histories.items():
        best_epoch = max(history, key=lambda h: h["valid"])
        gap = best_epoch["train"] - best_epoch["valid"]
        models.append(model_name)
        gaps.append(gap)

    colors = [PALETTE.get(m, "#4C72B0") for m in models]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(models, gaps, color=colors)
    ax.set_ylabel("Train accuracy - Validation accuracy")
    ax.set_title(f"Overfitting gap at best epoch (features={features})")
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, v in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center")
    fig.tight_layout()
    path = os.path.join(out_dir, "overfitting_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_diagnostics():
    """
    Load results/diagnostics_<run_name>.json for all runs, where run_name = <model>_<features>_<number>
    """
    diagnostics = []
    for path in glob.glob("results/diagnostics/diagnostics_*.json"):
        with open(path) as f:
            data = json.load(f)
        filename = os.path.basename(path)
        diagnostics.append({"file": filename, "diagnostics": data})
    return diagnostics


def plot_oversmoothing(out_dir):
    """
    Plot mean pairwise cosine similarity by layer

    Increasing cosine similarity with depth indicates
    increasing representation similarity / over-smoothing
    """
    diagnostics = load_diagnostics()
    if not diagnostics:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for item in diagnostics:
        filename = item["file"]
        smoothing = item["diagnostics"].get("smoothing", [])
        if not smoothing:
            continue
        layers = [r["layer"] for r in smoothing]
        cosine = [r["cosine_similarity"] for r in smoothing]
        # Extract model + layer count from filename
        label = filename.replace("diagnostics_", "").replace(".json", "")
        ax.plot(layers, cosine, marker="o", label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean pairwise cosine similarity")
    ax.set_title("Over-smoothing diagnostic")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, "over_smoothing.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_oversquashing(out_dir):
    """
    Plot mean gradient influence as a function of graph distance
    """
    diagnostics = load_diagnostics()
    if not diagnostics:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for item in diagnostics:
        filename = item["file"]
        squashing = item["diagnostics"].get("squashing", [])
        if not squashing:
            continue
        distance_values = {}
        for target_result in squashing:
            for d, influence in zip(target_result["distance"], target_result["mean_influence"]):
                if d not in distance_values:
                    distance_values[d] = []
                distance_values[d].append(influence)
        if not distance_values:
            continue
        distances = sorted(distance_values)
        mean_influence = [sum(distance_values[d]) / len(distance_values[d]) for d in distances]
        label = filename.replace("diagnostics_", "").replace(".json", "")
        ax.plot(distances, mean_influence, marker="o", label=label)

    ax.set_xlabel("Graph distance from target")
    ax.set_ylabel("Mean gradient influence")
    ax.set_title("Over-squashing diagnostic")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, "over_squashing.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    df = load_results()
    if df is None:
        return
    md_path = export_results_md(df)
    print(f"Results table (markdown) saved: {md_path}")

    print("=== Comparison table (run per model / feature type / number (=variant)) ===\n")
    print(df[["model", "features", "val_acc", "test_acc"]].to_string(index=False))

    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"\n=== Generating figures in {FIG_DIR}/ ===")
    for fn, args in [
        (plot_model_comparison, (df, FIG_DIR)), 
        (plot_ablation, (df, FIG_DIR)), 
        (plot_training_curves, (FIG_DIR,)), 
        (plot_overfitting_gap, (FIG_DIR,)),
        (plot_oversmoothing, (FIG_DIR,)),
        (plot_oversquashing, (FIG_DIR,))
    ]:
        path = fn(*args)
        if path:
            print(f"  saved {path}")


if __name__ == "__main__":
    main()
