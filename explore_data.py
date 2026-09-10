"""
"Days 3-4" step of the plan: explore OGBN-Arxiv before modeling.
Produces printed statistics AND a set of figures saved to
results/figures/exploration/.

Usage:
    python explore_data.py
"""

import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from torch_geometric.utils import degree

from src.data import load_arxiv

sns.set_theme(style="whitegrid")
FIG_DIR = "results/figures/exploration"


def plot_degree_distribution(deg, out_dir):
    """
    Log-log histogram of node degrees: citation networks are typically scale-free, so this is usually a straight-ish line on a log-log plot
    """
    deg_np = deg.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(deg_np, bins=60, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Node degree")
    axes[0].set_ylabel("Number of nodes")
    axes[0].set_title("Degree distribution (linear scale)")

    axes[1].hist(deg_np, bins=np.logspace(0, np.log10(max(deg_np.max(), 1)), 60), color="#DD8452", edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Node degree (log)")
    axes[1].set_ylabel("Number of nodes (log)")
    axes[1].set_title("Degree distribution (log-log scale)")

    fig.suptitle("OGBN-Arxiv — Degree distribution", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "degree_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_class_distribution(y, num_classes, out_dir):
    """
    Bar chart of the 40 class frequencies, sorted
    """
    counts = Counter(y.tolist())
    freqs = [counts.get(c, 0) for c in range(num_classes)]
    order = np.argsort(freqs)[::-1]
    sorted_freqs = np.array(freqs)[order]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(range(num_classes), sorted_freqs, color="#55A868")
    ax.set_xlabel("Class rank (most -> least frequent)")
    ax.set_ylabel("Number of papers")
    ax.set_title("OGBN-Arxiv — Class distribution (40 categories, sorted)")
    fig.tight_layout()
    path = os.path.join(out_dir, "class_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, sorted_freqs


def plot_degree_by_class(deg, y, num_classes, out_dir):
    """
    Boxplot of node degree per class: do some fields have much better
    connected papers than others?
    """
    deg_np = deg.numpy()
    y_np = y.numpy()

    # keep only the 15 largest classes for readability
    counts = Counter(y_np.tolist())
    top_classes = [c for c, _ in sorted(counts.items(), key=lambda x: -x[1])[:15]]

    data_per_class = [deg_np[y_np == c] for c in top_classes]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data_per_class, labels=[str(c) for c in top_classes], showfliers=False)
    ax.set_xlabel("Class id (15 largest classes)")
    ax.set_ylabel("Node degree")
    ax.set_title("Degree distribution per class (outliers hidden)")
    fig.tight_layout()
    path = os.path.join(out_dir, "degree_by_class.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _build_adjacency(data):
    """
    Plain Python adjacency list, built once and reused by both sampling modes below (cheap: O(E), done once per run)
    """
    adj = defaultdict(set)
    src, dst = data.edge_index[0].tolist(), data.edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].add(d)
        adj[d].add(s)
    return adj



def _sample_hub_neighborhood(adj, deg, num_nodes, seed):
    """BFS ("snowball") sample starting from a well-connected node, so the
    resulting subgraph is actually connected and shows a real local
    citation neighborhood -- unlike a uniform random sample, which would
    mostly be disconnected singletons on a graph this sparse."""
    rng = np.random.default_rng(seed)
    top_degree_nodes = np.argsort(deg.numpy())[::-1][:200]
    seed_node = int(rng.choice(top_degree_nodes))

    visited = {seed_node}
    frontier = [seed_node]
    while frontier and len(visited) < num_nodes:
        next_frontier = []
        for node in frontier:
            neighbors = list(adj[node])
            rng.shuffle(neighbors)
            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    next_frontier.append(n)
                    if len(visited) >= num_nodes:
                        break
            if len(visited) >= num_nodes:
                break
        frontier = next_frontier
    return list(visited)


def _sample_random_nodes(num_total_nodes, num_nodes, seed):
    """
    Uniform random sample of node ids, for contrast with the hub sample
    """
    rng = np.random.default_rng(seed)
    return rng.choice(num_total_nodes, size=num_nodes, replace=False).tolist()


def plot_graph_sample(data, deg, nodes, title, filename, out_dir, seed=42):
    """
    Draw an actual citation subgraph with networkx: nodes colored by class, sized by (full-graph) citation degree. 
    The full 169k-node graph can't be meaningfully plotted, so this shows a representative sample
    """
    node_set = set(nodes)
    src, dst = data.edge_index[0].tolist(), data.edge_index[1].tolist()
    edges = [(s, d) for s, d in zip(src, dst) if s in node_set and d in node_set and s != d]

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    y_np = data.y.numpy()
    node_classes = [int(y_np[n]) for n in G.nodes()]
    node_sizes = [20 + 4 * deg[n].item() for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=seed, k=0.4 if len(nodes) < 250 else 0.2)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.6)
    scatter = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_classes, cmap="tab20", node_size=node_sizes, linewidths=0.5, edgecolors="white")
    fig.colorbar(scatter, ax=ax, label="Class id", shrink=0.7)
    ax.set_title(f"{title}\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges shown | color = class, size = citation degree)")
    ax.axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_split_sizes(split_idx, out_dir):
    labels = ["train", "valid", "test"]
    sizes = [len(split_idx[s]) for s in labels]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.bar(labels, sizes, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Number of nodes")
    ax.set_title("Train / Validation / Test split sizes")
    for i, v in enumerate(sizes):
        ax.text(i, v + max(sizes) * 0.01, f"{v:,}", ha="center")
    fig.tight_layout()
    path = os.path.join(out_dir, "split_sizes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def explore():
    os.makedirs(FIG_DIR, exist_ok=True)
    data, split_idx, num_classes = load_arxiv()

    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)

    print("=== OGBN-Arxiv — general statistics ===")
    print(f"Number of nodes          : {num_nodes:,}")
    print(f"Number of edges          : {num_edges:,}  (graph made undirected)")
    print(f"Feature dimension        : {data.x.size(-1)}")
    print(f"Number of classes        : {num_classes}")
    print(f"Train / Val / Test       : {len(split_idx['train']):,} / {len(split_idx['valid']):,} / {len(split_idx['test']):,}")

    deg = degree(data.edge_index[0], num_nodes=num_nodes)
    print("\n=== Degrees (after symmetrization) ===")
    print(f"Mean degree              : {deg.mean().item():.2f}")
    print(f"Median degree            : {deg.median().item():.0f}")
    print(f"Max degree               : {deg.max().item():.0f}")
    print(f"Isolated nodes (degree 0): {(deg == 0).sum().item()}")

    print("\n=== Class distribution (10 most frequent) ===")
    counts = Counter(data.y.tolist())
    for cls, count in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"Class {cls:2d} : {count:5d} papers ({100 * count / num_nodes:.2f}%)")

    print("\n=== Homophily ===")
    # fraction of edges connecting two nodes of the same class
    src, dst = data.edge_index
    same_class = (data.y[src] == data.y[dst]).float().mean().item()
    print(f"Fraction of intra-class edges: {same_class:.3f}")
    print("(Closer to 1 = more homophilic graph -> structure should help a GNN more)")

    print(f"\n=== Generating figures in {FIG_DIR}/ ===")
    p1 = plot_degree_distribution(deg, FIG_DIR)
    print(f"  saved {p1}")
    p2, _ = plot_class_distribution(data.y, num_classes, FIG_DIR)
    print(f"  saved {p2}")
    p3 = plot_degree_by_class(deg, data.y, num_classes, FIG_DIR)
    print(f"  saved {p3}")
    p4 = plot_split_sizes(split_idx, FIG_DIR)
    print(f"  saved {p4}")

    print("\n=== Sampling and drawing the actual citation graph ===")
    print("(the full graph has 169k+ nodes and can't be meaningfully plotted, so we draw small representative samples instead)")
    adj = _build_adjacency(data)

    hub_nodes = _sample_hub_neighborhood(adj, deg, num_nodes=150, seed=42)
    p5 = plot_graph_sample(data, deg, hub_nodes, title="OGBN-Arxiv — citation neighborhood around a highly-cited paper", filename="graph_sample_hub.png", out_dir=FIG_DIR)
    print(f"  saved {p5}")

    random_nodes = _sample_random_nodes(num_nodes, 150, seed=42)
    p6 = plot_graph_sample(data, deg, random_nodes, title="OGBN-Arxiv — uniform random sample of papers (for contrast)", filename="graph_sample_random.png", out_dir=FIG_DIR)
    print(f"  saved {p6}")
    print("  (compare the two: the random sample is much sparser/more disconnected, which is typical of scale-free citation networks)")


if __name__ == "__main__":
    explore()
