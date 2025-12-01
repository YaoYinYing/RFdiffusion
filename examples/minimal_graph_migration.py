"""
Minimal example derived from the example suite to exercise both DGL and
PyTorch Geometric backends during the migration process.
"""

import torch
import dgl
from torch_geometric.data import Data


def build_graphs():
    src = torch.tensor([0, 1, 2, 2])
    dst = torch.tensor([1, 2, 0, 3])
    features = torch.arange(4, dtype=torch.float32).unsqueeze(-1)

    dgl_graph = dgl.graph((src, dst))
    dgl_graph.ndata["x"] = features

    pyg_graph = Data(x=features, edge_index=torch.stack([src, dst], dim=0))
    return dgl_graph, pyg_graph


def main():
    dgl_graph, pyg_graph = build_graphs()
    print("DGL graph:", dgl_graph)
    print("PyG graph:", pyg_graph)


if __name__ == "__main__":
    main()
