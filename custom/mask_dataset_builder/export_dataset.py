import argparse
from pathlib import Path

from .env_factory import make_mask_dataset_env
from .io_utils import ManifestWriter, frame_stem, prepare_task_output, write_mask, write_rgb
from .preview import overlay_mask
from .random_rollout import iter_random_rollout
from .render_pairs import render_pair
from .task_specs import TASK_MASK_SPECS, configured_task_names, get_task_mask_spec


def _time_step_scalar(value, default):
    return float(default if value is None else value)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export aligned RGB/mask pairs from DrQ-v2 tasks.')
    parser.add_argument('--output-dir',
                        type=Path,
                        required=True,
                        help='Directory where the dataset should be written.')
    parser.add_argument('--task',
                        action='append',
                        default=[],
                        help='Task name to export. Can be provided multiple times.')
    parser.add_argument('--all-configured-tasks',
                        action='store_true',
                        help='Export all tasks listed in cfgs/task/*.yaml.')
    parser.add_argument('--episodes-per-task', type=int, default=1)
    parser.add_argument('--max-steps-per-episode', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--render-height', type=int, default=180)
    parser.add_argument('--render-width', type=int, default=180)
    parser.add_argument('--action-repeat', type=int, default=1)
    parser.add_argument('--preview-every',
                        type=int,
                        default=0,
                        help='If > 0, save an overlay preview every N exported frames.')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Replace the per-task manifest if it already exists.')
    return parser.parse_args()


def determine_tasks(args):
    if args.task:
        return args.task
    if args.all_configured_tasks:
        tasks = configured_task_names()
        unsupported = sorted(set(tasks) - set(TASK_MASK_SPECS))
        if unsupported:
            joined = ', '.join(unsupported)
            raise ValueError(f'Configured tasks missing mask specs: {joined}')
        return tasks
    return sorted(TASK_MASK_SPECS)


def export_task(task_name, args, task_seed):
    spec = get_task_mask_spec(task_name)
    save_previews = args.preview_every > 0
    layout = prepare_task_output(args.output_dir,
                                 task_name,
                                 include_previews=save_previews)
    env = make_mask_dataset_env(task_name=task_name,
                                seed=task_seed,
                                action_repeat=args.action_repeat,
                                render_height=args.render_height,
                                render_width=args.render_width)
    exported_frames = 0
    with ManifestWriter(layout.manifest_path, overwrite=args.overwrite) as manifest:
        for rollout_frame in iter_random_rollout(
                env,
                num_episodes=args.episodes_per_task,
                max_steps_per_episode=args.max_steps_per_episode,
                seed=task_seed):
            pair = render_pair(env, spec)
            stem = frame_stem(rollout_frame.episode_index, rollout_frame.frame_index)
            image_path = layout.images_dir / f'{stem}.png'
            mask_path = layout.masks_dir / f'{stem}.png'
            write_rgb(image_path, pair.rgb)
            write_mask(mask_path, pair.mask)
            preview_path = None
            if save_previews and exported_frames % args.preview_every == 0:
                preview_path = layout.previews_dir / f'{stem}.png'
                write_rgb(preview_path, overlay_mask(pair.rgb, pair.mask))
            manifest.write({
                'task_name': task_name,
                'episode_index': rollout_frame.episode_index,
                'frame_index': rollout_frame.frame_index,
                'seed': task_seed,
                'image_path': str(image_path.relative_to(layout.task_dir)),
                'mask_path': str(mask_path.relative_to(layout.task_dir)),
                'preview_path': None if preview_path is None else str(
                    preview_path.relative_to(layout.task_dir)),
                'geom_ids': list(pair.geom_ids),
                'action': None if rollout_frame.action is None else
                rollout_frame.action.tolist(),
                'reward': _time_step_scalar(rollout_frame.time_step.reward, 0.0),
                'discount': _time_step_scalar(rollout_frame.time_step.discount,
                                              1.0),
                'mask_description': spec.description,
            })
            exported_frames += 1
    env.close()
    return exported_frames


def main():
    args = parse_args()
    tasks = determine_tasks(args)
    total_frames = 0
    for task_index, task_name in enumerate(tasks):
        task_seed = args.seed + task_index
        total_frames += export_task(task_name, args, task_seed)
    print(
        f'Exported {total_frames} RGB/mask pairs across {len(tasks)} task(s) into {args.output_dir}'
    )


if __name__ == '__main__':
    main()
