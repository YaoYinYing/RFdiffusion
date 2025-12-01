import torch
import dgl
from torch_geometric.data import Data


def test_graph_construction_parity():
    src = torch.tensor([0, 1, 2, 2])
    dst = torch.tensor([1, 2, 0, 3])
    features = torch.arange(4, dtype=torch.float32).unsqueeze(-1)

    dgl_graph = dgl.graph((src, dst))
    dgl_graph.ndata["x"] = features

    pyg_graph = Data(x=features, edge_index=torch.stack([src, dst], dim=0))

    assert dgl_graph.num_nodes() == pyg_graph.num_nodes
    assert dgl_graph.num_edges() == pyg_graph.num_edges

    dgl_edges = torch.stack(dgl_graph.edges(order="eid"), dim=0)
    pyg_edges = pyg_graph.edge_index
    assert torch.equal(dgl_edges, pyg_edges)
    assert torch.equal(dgl_graph.ndata["x"], pyg_graph.x)
