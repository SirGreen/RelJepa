"""RelJEPA: joint-embedding predictive pretraining for relational databases.

Reusable pieces (target view, objective, EMA, diagnostics). The training entry point lives in the
vendored TVE tree (third_party/task-vector-estimation/src/train_reljepa.py) so that every method
shares TVE's backbone, sampler and training loop.
"""
