"""
Generic training loop, shared by all 4 models.

Full-batch training: at the scale of OGBN-Arxiv (169k nodes, 128 features)
this comfortably fits in GPU memory (a few hundred MB). If you move to a
much larger graph, swap full-batch training for a
torch_geometric.loader.NeighborLoader (mini-batch + neighbor sampling).
"""

import copy
import os

import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from ogb.nodeproppred import Evaluator

from src.models import GCN, GraphSAGE, GAT

def train_epoch(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, split_idx, evaluator):
    model.eval()
    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)
    y_true = data.y.unsqueeze(1)
    results = {}
    for split in ["train", "valid", "test"]:
        idx = split_idx[split]
        results[split] = evaluator.eval({"y_true": y_true[idx], "y_pred": y_pred[idx]})["acc"]
    return results


@torch.no_grad()
def get_embeddings(model, x, edge_index):
    """
    Reconstruct hidden representations WITHOUT modifying model.forward()
    -> for GCN, GraphSAGE, GAT only (MLP has no hidden representations + ResGCN is a special case of GCN)
    Heavy to repeat but intentional to avoid slowing down training with extra forward() calls.
    Only used after training
    """
    embeddings = [x.detach()]
    h = x
    # Hidden GCN layers
    for conv, bn in zip(model.convs[:-1], model.bns):
        h = conv(h, edge_index)
        h = bn(h)
        h = F.relu(h)
        embeddings.append(h.detach())
    # Final GCN layer
    h = model.convs[-1](h, edge_index)
    embeddings.append(h.detach())
    return embeddings


@torch.no_grad()
def oversmoothing_metrics(embeddings, num_pairs=10000):
    """
    Compute cheap post-training over-smoothing diagnostics
    Higher cosine similarity + lower variance => stronger representation collapse / over-smoothing
    """
    results = []
    for layer, h in enumerate(embeddings):
        n = h.size(0)

        i = torch.randint(0, n, (num_pairs,), device=h.device)
        j = torch.randint(0, n, (num_pairs,), device=h.device)
        h_i = F.normalize(h[i], p=2, dim=-1)
        h_j = F.normalize(h[j], p=2, dim=-1)
        cosine = (h_i * h_j).sum(dim=-1).mean().item()

        mean = h.mean(dim=0, keepdim=True)
        variance = ((h - mean).pow(2).sum(dim=-1)).mean().item()

        results.append({"layer": layer, "cosine_similarity": cosine, "embedding_variance": variance})
    return results


def oversquashing_diagnostic(model, data, target_nodes=None, max_distance=5, n_targets=20):
    """
    Post-training over-squashing diagnostic

    For a few target nodes, measure how strongly their final hidden representation depends on input features at each graph distance
    This is intentionally expensive but is run ONLY once after training
    """
    model.eval()
    x = data.x.detach().clone()
    edge_index = data.edge_index

    # We need gradients through x
    x.requires_grad_(True)
    num_nodes = x.size(0)
    if target_nodes is None:
        target_nodes = torch.randperm(num_nodes, device=x.device)[:n_targets]

    all_results = []
    for target in target_nodes:
        # ---- Reconstruct final hidden representation ----
        h = x
        for conv, bn in zip(model.convs[:-1], model.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)

        # Representation immediately before classification
        h = model.convs[-1](h, edge_index)
        target_embedding = h[target]

        # Scalar objective
        scalar = target_embedding.norm()
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        scalar.backward()

        # || dh_target / dx_u ||
        influence = x.grad.norm(dim=1)

        # ---- Compute graph distances from target ----
        distances = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)
        distances[target] = 0
        frontier = torch.tensor([target], device=x.device, dtype=torch.long)
        for d in range(1, max_distance + 1):
            # Nodes adjacent to current frontier
            src = edge_index[0]
            dst = edge_index[1]
            mask = torch.isin(src, frontier)

            neighbors = dst[mask]
            # Remove already visited nodes
            neighbors = neighbors[distances[neighbors] == -1]
            if neighbors.numel() == 0:
                break

            neighbors = torch.unique(neighbors)
            distances[neighbors] = d
            frontier = neighbors

        # ---- Aggregate influence by distance ----
        target_result = {"target": int(target.item()), "distance": [], "mean_influence": [], "max_influence": [], "num_nodes": []}
        for d in range(0, max_distance + 1):
            mask = distances == d
            if mask.sum() == 0:
                continue
            values = influence[mask]
            target_result["distance"].append(d)
            target_result["mean_influence"].append(values.mean().item())
            target_result["max_influence"].append(values.max().item())
            target_result["num_nodes"].append(int(mask.sum().item()))
        all_results.append(target_result)
    return all_results


def run_training(model, data, split_idx, epochs=300, lr=0.01, weight_decay=0.0, patience=50, verbose=True, num_pairs=10000, max_distance=5, n_targets=20):
    """
    Train the model, keep the best checkpoint on validation accuracy,
    and return final model, best_val_acc, best_test_acc, history, diagnostics (if in a GNN)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    evaluator = Evaluator(name="ogbn-arxiv")

    best_val = 0.0
    best_test = 0.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, split_idx["train"], optimizer)
        results = evaluate(model, data, split_idx, evaluator)
        history.append({"epoch": epoch, "loss": loss, **results})

        if results["valid"] > best_val:
            best_val = results["valid"]
            best_test = results["test"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | loss {loss:.4f} | train {results['train']:.4f} | val {results['valid']:.4f} | test {results['test']:.4f}")

        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)

    # ============================================================
    # Post training diagnostics (GCN only)
    # ============================================================
    diagnostics = None
    if isinstance(model, GCN) or isinstance(model, GraphSAGE) or isinstance(model, GAT):
        diagnostics = {}
        print("\n=== Post-training diagnostics ===")

        # Recalculate embeddings layer-by-layer
        model.eval()
        embeddings = get_embeddings(model, data.x, data.edge_index)

        # Over-smoothing
        smoothing = oversmoothing_metrics(embeddings, num_pairs=num_pairs)
        print("\nOver-smoothing:")
        for r in smoothing:
            print(f"Layer {r['layer']:02d} | cosine = {r['cosine_similarity']:.4f} | variance = {r['embedding_variance']:.4f}")

        # Over-squashing
        print("\nOver-squashing:")
        print(f"Computing gradient influence for {n_targets} target nodes...")
        squashing = oversquashing_diagnostic(model, data, target_nodes=None, max_distance=max_distance, n_targets=n_targets)

        for r in squashing:
            print(f"\nTarget node {r['target']}")
            for d, mean_inf, n in zip(r["distance"], r["mean_influence"], r["num_nodes"]):
                print(f"  distance {d}: mean influence = {mean_inf:.6e} (n={n})")

        diagnostics = {"smoothing": smoothing, "squashing": squashing}
    return model, best_val, best_test, history, diagnostics


def save_checkpoint(model, path, config=None, val_acc=None, test_acc=None):
    """
    Save a trained model plus its config/metrics to `path` (.pt file)

    The checkpoint contains everything needed to reload and re-evaluate the model later: state_dict, hyperparameters, and final scores
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config or {}, "val_acc": val_acc, "test_acc": test_acc}, path)


def load_checkpoint(path, map_location=None):
    """
    Load a checkpoint saved with save_checkpoint(), returns the dict

    use build_model(**checkpoint['config']) + model.load_state_dict(...) to reconstruct a usable model
    """
    return torch.load(path, map_location=map_location)
