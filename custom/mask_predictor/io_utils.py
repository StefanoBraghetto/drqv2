import json
from pathlib import Path

import torch


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, data):
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + '\n',
                    encoding='utf-8')


def append_jsonl(path, data):
    path = Path(path)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(data, sort_keys=True, default=str) + '\n')


def save_checkpoint(path,
                    model,
                    optimizer,
                    epoch,
                    config,
                    metrics,
                    extra_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None if optimizer is None else optimizer.state_dict(),
        'config': config,
        'metrics': metrics,
        'extra_state': extra_state or {},
    }
    torch.save(payload, path)


def load_checkpoint(path, map_location='cpu'):
    return torch.load(path, map_location=map_location, weights_only=False)
