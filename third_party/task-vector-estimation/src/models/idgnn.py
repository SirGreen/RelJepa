"""ID-GNN recommendation head on TVE's backbone (port of RelBench v1.1.0 examples/idgnn_link.py).

Same backbone as `Baseline` (so pretrained backbone checkpoints load unchanged); an identity
embedding is added to the seed (user) nodes and the linear head scores every destination-table
node (article) in the seed's sampled subgraph.
"""
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from src.models.baseline import Baseline


class IDGNN(Baseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        channels = kwargs["conv_kwargs"]["channels"]
        self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.id_awareness_emb.reset_parameters()

    def forward_dst_readout(self, batch: HeteroData, src_table: NodeType, dst_table: NodeType) -> Tensor:
        seed_time = batch[src_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # ID-awareness: mark the seed node of every subgraph.
        x_dict[src_table][: seed_time.size(0)] += self.id_awareness_emb.weight
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        return self.head(x_dict[dst_table])
