"""Linear-probing helper: the backbone is frozen and stays in eval mode; the head trains."""
import torch

from src.helpers.reljepa_ft_helper import freeze_backbone


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5))
        self.temporal_encoder = torch.nn.Linear(4, 4)
        self.gnn = torch.nn.Linear(4, 4)
        self.embedding_dict = torch.nn.ModuleDict()
        self.head = torch.nn.Linear(4, 1)


def test_freeze_backbone_keeps_eval_mode_and_trains_head_only():
    model = freeze_backbone(Toy())
    model.train()
    assert not model.encoder.training and not model.gnn.training and not model.temporal_encoder.training
    assert model.head.training
    assert all(not p.requires_grad for m in (model.encoder, model.temporal_encoder, model.gnn) for p in m.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    x = torch.randn(8, 4)
    out = model.head(model.gnn(model.temporal_encoder(model.encoder(x))))
    out.sum().backward()
    assert model.head.weight.grad is not None and model.gnn.weight.grad is None
