from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_BACKGROUND_GEOM_FRAGMENTS = (
    'floor',
    'ground',
    'wall',
    'arena',
    'track',
    'rail',
    'ramp',
    'target_ground',
)


@dataclass(frozen=True)
class TaskMaskSpec:
    task_name: str
    description: str
    include_all_geoms: bool = False
    include_geom_name_fragments: tuple[str, ...] = ()
    include_body_name_fragments: tuple[str, ...] = ()
    exclude_geom_name_fragments: tuple[str, ...] = ()


def _make_spec(task_name, description, **kwargs):
    return task_name, TaskMaskSpec(task_name=task_name,
                                   description=description,
                                   **kwargs)


def _same_spec(task_names, description, **kwargs):
    return [_make_spec(task_name, description, **kwargs) for task_name in task_names]


TASK_MASK_SPECS = dict(
    _same_spec(
        (
            'acrobot_swingup',
            'cartpole_balance',
            'cartpole_balance_sparse',
            'cartpole_swingup',
            'cartpole_swingup_sparse',
            'cheetah_run',
            'finger_spin',
            'finger_turn_easy',
            'finger_turn_hard',
            'hopper_hop',
            'hopper_stand',
            'humanoid_run',
            'humanoid_stand',
            'humanoid_walk',
            'pendulum_swingup',
            'quadruped_run',
            'quadruped_walk',
            'reacher_easy',
            'reacher_hard',
            'walker_run',
            'walker_stand',
            'walker_walk',
        ),
        description='Mask all task-relevant visible geoms while excluding common background geometry.',
        include_all_geoms=True,
        exclude_geom_name_fragments=DEFAULT_BACKGROUND_GEOM_FRAGMENTS,
    ) + _same_spec(
        ('cup_catch', ),
        description='Mask the cup and ball used by the catch task.',
        include_geom_name_fragments=('cup', 'ball'),
    ) + _same_spec(
        ('reach_duplo', ),
        description='Mask the manipulator fingertips and the Duplo block.',
        include_geom_name_fragments=('duplo', 'finger', 'thumb', 'hand',
                                     'gripper', 'palm', 'target'),
        include_body_name_fragments=('duplo', 'finger', 'thumb', 'hand',
                                     'gripper', 'palm'),
        exclude_geom_name_fragments=('floor', 'table', 'ground', 'arena'),
    ))


def get_task_mask_spec(task_name):
    try:
        return TASK_MASK_SPECS[task_name]
    except KeyError as exc:
        available = ', '.join(sorted(TASK_MASK_SPECS))
        raise KeyError(f'No mask spec registered for {task_name!r}. '
                       f'Available task specs: {available}') from exc


def configured_task_names(config_dir=None):
    config_root = Path(config_dir or Path(__file__).resolve().parents[2] / 'cfgs' /
                       'task')
    task_names = []
    for path in sorted(config_root.glob('*.yaml')):
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict) and 'task_name' in data:
            task_names.append(data['task_name'])
    return task_names

