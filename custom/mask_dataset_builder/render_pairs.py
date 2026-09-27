from dataclasses import dataclass

import numpy as np

from .task_specs import TaskMaskSpec


@dataclass(frozen=True)
class RenderedPair:
    rgb: np.ndarray
    mask: np.ndarray
    geom_ids: tuple[int, ...]


def _matches(name, fragments):
    lowered_name = name.lower()
    return any(fragment.lower() in lowered_name for fragment in fragments)


def resolve_geom_ids(model, task_spec: TaskMaskSpec):
    selected = set()
    geom_names = {}
    for geom_id in range(model.ngeom):
        name = model.id2name(geom_id, 'geom')
        if name:
            geom_names[geom_id] = name
            if task_spec.include_all_geoms:
                selected.add(geom_id)
            if _matches(name, task_spec.include_geom_name_fragments):
                selected.add(geom_id)
    if task_spec.include_body_name_fragments and hasattr(model, 'geom_bodyid'):
        body_name_by_id = {}
        for body_id in range(getattr(model, 'nbody', 0)):
            body_name = model.id2name(body_id, 'body')
            if body_name:
                body_name_by_id[body_id] = body_name
        for geom_id, geom_name in geom_names.items():
            del geom_name
            body_id = int(model.geom_bodyid[geom_id])
            body_name = body_name_by_id.get(body_id)
            if body_name and _matches(body_name,
                                      task_spec.include_body_name_fragments):
                selected.add(geom_id)
    for geom_id, geom_name in geom_names.items():
        if _matches(geom_name, task_spec.exclude_geom_name_fragments):
            selected.discard(geom_id)
    return tuple(sorted(selected))


def _geom_object_type_id():
    try:
        import mujoco
    except ImportError:
        return None
    return int(mujoco.mjtObj.mjOBJ_GEOM)


def segmentation_to_mask(segmentation, geom_ids):
    geom_ids = tuple(int(geom_id) for geom_id in geom_ids)
    if not geom_ids:
        return np.zeros(segmentation.shape[:2], dtype=bool)
    if segmentation.ndim != 3:
        raise ValueError(
            f'Expected segmentation image with 3 dimensions, got {segmentation.shape!r}'
        )
    if segmentation.shape[2] == 1:
        return np.isin(segmentation[..., 0], geom_ids)
    if segmentation.shape[2] < 2:
        raise ValueError(
            f'Expected segmentation image with at least 2 channels, got {segmentation.shape!r}'
        )
    object_ids = segmentation[..., 0]
    object_types = segmentation[..., 1]
    mask = np.isin(object_ids, geom_ids)
    geom_object_type = _geom_object_type_id()
    if geom_object_type is not None:
        mask &= object_types == geom_object_type
    return mask


def render_pair(env, task_spec):
    rgb = env.render_rgb()
    segmentation = env.render_segmentation()
    geom_ids = resolve_geom_ids(env.physics.model, task_spec)
    if not geom_ids:
        raise ValueError(f'No visible geoms matched mask spec for {task_spec.task_name}')
    mask = segmentation_to_mask(segmentation, geom_ids)
    if rgb.shape[:2] != mask.shape[:2]:
        raise ValueError('RGB and mask shapes do not align')
    return RenderedPair(rgb=rgb, mask=mask.astype(bool), geom_ids=geom_ids)
