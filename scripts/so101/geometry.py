"""Calibrated joint bounds and MuJoCo collision checks for policy replay."""

import math

ENCODER_STEP = math.tau / 4095  # HX-30HM


def configure_rollout(env, limits, names):
    """Use physical actuator travel while retaining the policy's trained observation scale."""
    for i, name in enumerate(names):
        low = max(env.limits[i, 0], limits[name]['min_rad']) + ENCODER_STEP
        high = min(env.limits[i, 1], limits[name]['max_rad']) - ENCODER_STEP
        if not low < high:
            raise ValueError(f'No usable actuator range for {name}')
        env.model.actuator_ctrlrange[env.actuators[i]] = (low, high)


def check_pose(env, pose, margin):
    import mujoco
    import numpy as np

    q = np.asarray(pose)
    if (q.shape != (6,) or not np.isfinite(q).all()
            or np.any(q < env.limits[:, 0]) or np.any(q > env.limits[:, 1])):
        raise ValueError('Replay pose outside model limits')
    env.data.qpos[env.addresses] = q
    mujoco.mj_forward(env.model, env.data)
    if env._collision():
        raise ValueError('Replay pose in collision')
    for geom in env.arm_geoms:
        gap = mujoco.mj_geomDistance(env.model, env.data, env.block_geom, int(geom), margin, None)
        if not math.isfinite(gap) or gap < margin:
            raise ValueError('Replay pose violates obstacle clearance')


def check_segment(env, start, end, margin):
    import numpy as np

    delta = np.subtract(end, start)
    steps = max(1, math.ceil(np.max(np.abs(delta)) / ENCODER_STEP))
    for i in range(1, steps + 1):
        check_pose(env, start + delta * (i / steps), margin)
