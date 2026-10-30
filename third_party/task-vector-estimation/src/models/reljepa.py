"""RelJEPA model: context encoder f_θ, EMA target encoder f_ξ, predictor g_φ.

The context encoder is exactly TVE's / RelBench's backbone (HeteroEncoder -> temporal encoding ->
heterogeneous GraphSAGE) and keeps TVE's attribute names (`encoder`, `temporal_encoder`, `gnn`,
`embedding_dict`), so `save_backbone_checkpoint` / `load_backbone_checkpoint` work unchanged and
fine-tuning starts from f_θ.
"""
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict

from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn.inits import reset
from torch_geometric.typing import NodeType

from src.models.components import HeteroEncoder, HeteroTemporalEncoder, HeteroModel
from src.models.tve import TVE

from reljepa.objective import Predictor
from reljepa.target_view import make_target_view, pool_targets


def encode(modules: Dict[str, torch.nn.Module], batch: HeteroData, src_table: NodeType) -> Dict[str, Tensor]:
    """Backbone forward pass shared by the context and target encoders (same as TVE.forward minus the head)."""
    seed_time = batch[src_table].seed_time
    x_dict = modules["encoder"](batch.tf_dict)
    rel_time_dict = modules["temporal_encoder"](seed_time, batch.time_dict, batch.batch_dict)
    for node_type, rel_time in rel_time_dict.items():
        x_dict[node_type] = x_dict[node_type] + rel_time
    for node_type, embedding in modules["embedding_dict"].items():
        x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
    return modules["gnn"](x_dict, batch.edge_index_dict)


class RelJEPA(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        schema_graph: Data,
        schema_line_graph: Data,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        target_types: List[str],
        hetero_encoder_cls: Type[torch.nn.Module] = None,
        hetero_encoder_kwargs: Dict[str, Any] = None,
        temporal_encoder_kwargs: Dict[str, Any] = None,
        conv_cls: Type[torch.nn.Module] = None,
        conv_kwargs: Dict[str, Any] = None,
        shallow_list: List[NodeType] = [],
        mask_rate: float = 0.15,
        tve_out_channels: Optional[int] = None,
        use_ema: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.mask_rate = mask_rate
        self.target_types = list(target_types)
        self.use_ema = use_ema

        # Context encoder f_θ (identical construction to TVE / Baseline).
        self.encoder = HeteroEncoder(
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats_dict,
            torch_frame_model_cls=hetero_encoder_cls,
            torch_frame_model_kwargs=hetero_encoder_kwargs,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]],
            **temporal_encoder_kwargs,
        )
        self.gnn = HeteroModel(
            node_types=data.node_types,
            edge_types=data.edge_types,
            conv_cls=conv_cls,
            text_embedder=hetero_encoder_kwargs["text_embedder"],
            **conv_kwargs,
        )
        self.embedding_dict = ModuleDict(
            {node: Embedding(data.num_nodes_dict[node], conv_kwargs["channels"]) for node in shallow_list}
        )

        channels = conv_kwargs["channels"]
        self.predictor = Predictor(channels, self.target_types)
        # Optional TVE head on z_ctx for RelJEPA+TVE.
        self.tve_head = torch.nn.Linear(channels, tve_out_channels) if tve_out_channels else None

        self.schema_graph = schema_graph
        self.schema_line_graph = schema_line_graph
        self.reset_parameters()

        # EMA target encoder f_ξ: a frozen copy updated by `reljepa.ema.ema_update`.
        # Without EMA (SIGReg / no-anti-collapse variants) the context modules are shared.
        if use_ema:
            self.target = ModuleDict({k: deepcopy(m) for k, m in self.context_modules().items()})
            for p in self.target.parameters():
                p.requires_grad_(False)
        else:
            self.target = None

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.temporal_encoder)
        reset(self.gnn)
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.tve_head is not None:
            reset(self.tve_head)

    def context_modules(self) -> Dict[str, torch.nn.Module]:
        return {
            "encoder": self.encoder,
            "temporal_encoder": self.temporal_encoder,
            "gnn": self.gnn,
            "embedding_dict": self.embedding_dict,
        }

    def target_modules(self) -> Dict[str, torch.nn.Module]:
        return dict(self.target.items()) if self.target is not None else self.context_modules()

    def mask_input(self, batch: HeteroData) -> HeteroData:
        """TVE's input corruption (15% cell masking/shuffling during training), reused for parity."""
        return TVE.mask_input(self, batch)

    def forward(self, ctx_batch: HeteroData, tgt_batch: HeteroData, src_table: NodeType, delta_seconds: int):
        batch_size = ctx_batch[src_table].batch_size

        # Context view G_{<=t}(v) -> z_ctx.
        x_ctx = encode(self.context_modules(), self.mask_input(ctx_batch), src_table)
        z_ctx = x_ctx[src_table][:batch_size]
        preds = self.predictor(z_ctx)

        # Target view T(v, t, Δ) -> pooled targets. EMA: no gradient (stop-gradient).
        tview, _ = make_target_view(tgt_batch, src_table, delta_seconds)
        modules = self.target_modules()
        if self.target is not None:
            was_training = [m.training for m in modules.values()]
            for m in modules.values():
                m.eval()
            with torch.no_grad():
                h_dict = encode(modules, tview, src_table)
            for m, flag in zip(modules.values(), was_training):
                m.train(flag)
        else:
            h_dict = encode(modules, tview, src_table)
        means, counts = pool_targets(h_dict, tview, src_table, self.target_types)

        out = {"z_ctx": z_ctx, "preds": preds, "targets": means, "counts": counts}
        if self.tve_head is not None:
            out["tve"] = self.tve_head(z_ctx)
        return out
