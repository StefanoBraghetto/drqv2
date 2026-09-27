import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .transforms import ImageMaskTransform


def _ensure_paths(paths):
    if isinstance(paths, (str, Path)):
        return [paths if isinstance(paths, Path) else Path(paths)]
    return [path if isinstance(path, Path) else Path(path) for path in paths]


def list_task_dirs(dataset_root):
    task_dirs = []
    for root in _ensure_paths(dataset_root):
        task_dirs.extend(path for path in root.iterdir() if path.is_dir())
    return sorted(task_dirs)


def list_manifest_paths(dataset_root, tasks=None):
    task_filter = None if tasks is None else set(tasks)
    manifests = []
    for task_dir in list_task_dirs(dataset_root):
        if task_filter is not None and task_dir.name not in task_filter:
            continue
        manifest_path = task_dir / 'manifest.jsonl'
        if manifest_path.exists():
            manifests.append(manifest_path)
    return manifests


def load_manifest_records(dataset_root, tasks=None):
    records = []
    for manifest_path in list_manifest_paths(dataset_root, tasks=tasks):
        task_dir = manifest_path.parent
        with manifest_path.open(encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                record['task_dir'] = str(task_dir)
                record['image_abspath'] = str(task_dir / record['image_path'])
                record['mask_abspath'] = str(task_dir / record['mask_path'])
                records.append(record)
    return records


def split_records(records, split='train', val_fraction=0.1, seed=1):
    if split == 'all':
        return list(records)
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f'val_fraction must be in [0, 1), got {val_fraction}')
    if not records:
        return list(records)
    if val_fraction == 0.0:
        return list(records) if split == 'train' else []
    by_task = {}
    for index, record in enumerate(records):
        by_task.setdefault(record['task_name'], []).append((index, record))
    train_records = []
    val_records = []
    rng = random.Random(seed)
    for task_name, task_entries in sorted(by_task.items()):
        del task_name
        task_entries = list(task_entries)
        rng.shuffle(task_entries)
        val_count = max(1, int(round(len(task_entries) * val_fraction)))
        if val_count >= len(task_entries):
            val_count = len(task_entries) - 1
        if val_count <= 0:
            train_records.extend(record for _, record in sorted(task_entries))
            continue
        val_subset = task_entries[:val_count]
        train_subset = task_entries[val_count:]
        train_records.extend(record for _, record in sorted(train_subset))
        val_records.extend(record for _, record in sorted(val_subset))
    if split == 'train':
        return train_records
    if split == 'val':
        return val_records
    raise ValueError(f'Unknown split: {split}')


class MaskDataset(Dataset):
    def __init__(self,
                 dataset_roots,
                 split='train',
                 tasks=None,
                 val_fraction=0.1,
                 seed=1,
                 transform=None,
                  max_samples=None,
                 image_size=180):
        records = load_manifest_records(dataset_roots, tasks=tasks)
        self.records = split_records(records,
                                     split=split,
                                     val_fraction=val_fraction,
                                     seed=seed)
        if max_samples is not None:
            self.records = self.records[:max_samples]
        self.transform = transform or ImageMaskTransform(train=(split == 'train'),
                                                        image_size=image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record['image_abspath']).convert('RGB')
        mask = Image.open(record['mask_abspath']).convert('L')
        image_tensor, mask_tensor = self.transform(image, mask)
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'task_name': record['task_name'],
            'image_path': record['image_abspath'],
            'mask_path': record['mask_abspath'],
            'episode_index': record.get('episode_index', -1),
            'frame_index': record.get('frame_index', -1),
        }
