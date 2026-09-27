from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskDatasetEnv:
    env: object
    raw_env: object
    task_name: str
    domain: str
    task: str
    camera_id: object
    render_height: int
    render_width: int
    action_repeat: int

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def action_spec(self):
        return self.env.action_spec()

    def close(self):
        close_fn = getattr(self.env, 'close', None)
        if close_fn is not None:
            close_fn()

    @property
    def physics(self):
        return self.raw_env.physics

    def render_rgb(self):
        return self.physics.render(height=self.render_height,
                                   width=self.render_width,
                                   camera_id=self.camera_id)

    def render_segmentation(self):
        return self.physics.render(height=self.render_height,
                                   width=self.render_width,
                                   camera_id=self.camera_id,
                                   segmentation=True)


def make_mask_dataset_env(task_name,
                          seed,
                          action_repeat=1,
                          render_height=84,
                          render_width=84,
                          camera_id=None):
    import dmc

    raw_env, pixels_key, is_classic, domain, task = dmc.load_raw(task_name, seed)
    env = dmc.ActionDTypeWrapper(raw_env, np.float32)
    env = dmc.ActionRepeatWrapper(env, action_repeat)
    env = dmc.action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    if camera_id is None:
        camera_id = dmc.default_camera_id(domain) if is_classic else pixels_key
    return MaskDatasetEnv(env=env,
                          raw_env=raw_env,
                          task_name=task_name,
                          domain=domain,
                          task=task,
                          camera_id=camera_id,
                          render_height=render_height,
                          render_width=render_width,
                          action_repeat=action_repeat)
