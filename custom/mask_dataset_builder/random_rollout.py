from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RolloutFrame:
    episode_index: int
    frame_index: int
    time_step: object
    action: np.ndarray | None


def sample_random_action(action_spec, rng):
    minimum = np.asarray(action_spec.minimum, dtype=np.float32)
    maximum = np.asarray(action_spec.maximum, dtype=np.float32)
    action = rng.uniform(low=minimum, high=maximum, size=action_spec.shape)
    return action.astype(action_spec.dtype, copy=False)


def iter_random_rollout(env, num_episodes, max_steps_per_episode, seed):
    rng = np.random.default_rng(seed)
    for episode_index in range(num_episodes):
        time_step = env.reset()
        yield RolloutFrame(episode_index=episode_index,
                           frame_index=0,
                           time_step=time_step,
                           action=None)
        frame_index = 0
        while frame_index < max_steps_per_episode and not time_step.last():
            action = sample_random_action(env.action_spec(), rng)
            time_step = env.step(action)
            frame_index += 1
            yield RolloutFrame(episode_index=episode_index,
                               frame_index=frame_index,
                               time_step=time_step,
                               action=action)
