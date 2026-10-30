"""Add our task names to TVE's data/dataset_task.json (idempotent).

TVE's `infer_dataset_task` only accepts task names listed in that file, which ships inside TVE's
released data.zip. Run inside the container from the TVE root:
  python /workspace/scripts/setup/register_tasks.py
"""
import json

NEW_TASKS = {
    "rel-hm": ["user-pt-tve-1-hop", "user-pt-ssl", "item-pt-tve-1-hop", "item-pt-ssl"],
}

path = "data/dataset_task.json"
with open(path) as f:
    tasks = json.load(f)
for dataset, names in NEW_TASKS.items():
    for name in names:
        if name not in tasks[dataset]:
            tasks[dataset].append(name)
with open(path, "w") as f:
    json.dump(tasks, f, indent=4)
print({k: v for k, v in tasks.items() if k in NEW_TASKS})
