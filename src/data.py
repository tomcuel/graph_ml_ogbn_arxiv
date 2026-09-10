"""
Loading the OGBN-Arxiv dataset.

OGB already provides:
- nodes (papers) with a 128-dim content feature vector (built from title/abstract)
- edges (citations), directed (A cites B)
- labels (40 categories)
- train / valid / test splits (based on publication year)

We make the graph undirected, which is standard practice on this dataset:
a paper is informed both by what it cites AND by what cites it.
"""

import torch
from ogb.nodeproppred import PygNodePropPredDataset


# --- PyTorch >= 2.6 / OGB compatibility fix -------------------------------
# Since PyTorch 2.6, torch.load() defaults to weights_only=True, but the `ogb` package internally calls torch.load() 
# on its own preprocessed dataset file (not a plain state_dict), which contains a few PyG classes that aren't allowlisted by default. 
# This is a known ogb/PyTorch version mismatch, not a corrupted file. 
# We explicitly allowlist those classes instead of disabling weights_only globally.
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except ImportError:
    # older torch_geometric versions don't need this at all
    pass
# ---------------------------------------------------------------------------


def load_arxiv(root="./data"):
    """
    Load OGBN-Arxiv and return (data, split_idx, num_classes)
    """
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    # Undirected graph: add reverse edges then deduplicate
    edge_index = data.edge_index
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(edge_index, dim=1)

    # data.y has shape (N, 1) -> reshape to (N,) for F.nll_loss
    data.y = data.y.squeeze(1)

    return data, split_idx, dataset.num_classes
