"""Replay a simulated far-side candidate that the planner rejects as short."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts import PROJECT_ROOT
from scripts.planning.plan_reach import (
    DEFAULT_CHECKPOINT,
    PPO,
    RelativePlanningEnv,
    candidate_targets,
)
from scripts.simulation.reaching_env import JOINT_NAMES
from scripts.so101.geometry import configure_rollout


def read_current_joints(port, calibration_path, mapping_path, motion_config, home):
    """Read follower feedback without sending position or torque commands."""
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.hiwonder import HiwonderMotorsBus, MODEL_RESOLUTION

    from scripts.so101.calibration import connected_reader, load_calibration, load_mapping
    from scripts.so101.hiwonder_control import HiwonderArm

    rows, digest = load_calibration(
        calibration_path, JOINT_NAMES, MODEL_RESOLUTION['hx30hm'] - 1
    )
    mapping = load_mapping(
        mapping_path, digest, rows, home,
        2 * np.pi / (MODEL_RESOLUTION['hx30hm'] - 1),
    )
    motors = {
        name: Motor(
            row['id'], 'hx30hm',
            MotorNormMode.RANGE_0_100 if name == 'gripper' else MotorNormMode.DEGREES,
        )
        for name, row in rows.items()
    }
    bus = HiwonderMotorsBus(
        port=port,
        motors=motors,
        calibration={name: MotorCalibration(**row) for name, row in rows.items()},
    )
    arm = HiwonderArm(bus, rows, mapping, motion_config)
    with connected_reader(bus, rows, exclusive=True):
        positions, velocities, moving = arm.sample(enforce_speed=False)
    print('Current follower pose (rad):', dict(zip(JOINT_NAMES, np.round(positions, 6))))
    print('Follower moving:', moving)
    return np.asarray(positions, dtype=float)


def find_short_candidate(env, policy, joints, obstacle_xy, margin, home_tolerance):
    """Return a simulated far-side rollout whose gripper stops short."""
    start = env.place_planning_start(joints, obstacle_xy, home_tolerance)
    direction = np.asarray(obstacle_xy, dtype=float) - start[:2]
    direction_length = np.linalg.norm(direction)
    if direction_length == 0:
        raise RuntimeError('The obstacle center matches the starting gripper position')
    direction /= direction_length
    targets = candidate_targets(env, start, obstacle_xy, direction, margin, seed=0)
    far_edge = np.abs(direction) @ env.obstacle_half_size[:2]

    for far, target in targets:
        if not far:
            continue
        try:
            observation, _ = env.reset_for_planning(
                joint_positions=joints,
                target_position=target,
                obstacle_xy=obstacle_xy,
                home_tolerance=home_tolerance,
            )
        except ValueError:
            continue

        env.min_gap = 2.0
        trajectory = [env.data.qpos[env.addresses].copy()]
        while True:
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, done, timeout, info = env.step(action)
            trajectory.append(env.data.qpos[env.addresses].copy())
            if info['collision'] or info['unsafe'] or timeout:
                break
            if done:
                tip = env.data.site_xpos[env.tip, :2]
                if (tip - obstacle_xy) @ direction <= far_edge:
                    return target, trajectory
                break

    raise RuntimeError('No rejected far-side candidate stopped short on this run')


def preview(env, target, trajectory, obstacle_xy, delay):
    """Animate a captured joint trajectory in the MuJoCo viewer."""
    import mujoco

    env.render_mode = 'human'
    env._set_block(obstacle_xy)
    env.data.mocap_pos[env.mocap] = target
    env.render()
    if env._viewer is None or not env._viewer.is_running():
        raise RuntimeError('Could not open the MuJoCo viewer')

    try:
        for positions in trajectory:
            if not env._viewer.is_running():
                break
            with env._viewer.lock():
                env.data.qpos[env.addresses] = positions
                mujoco.mj_forward(env.model, env.data)
                env._viewer.sync()
            time.sleep(delay)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    pose_source = parser.add_mutually_exclusive_group(required=True)
    pose_source.add_argument('--joints', type=float, nargs=6, metavar=tuple(JOINT_NAMES))
    pose_source.add_argument('--read-current', action='store_true',
                             help='Read the current Hiwonder follower encoders')
    parser.add_argument('--port', default='COM5')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--mapping', type=Path,
                        default=PROJECT_ROOT / 'scripts/hardware/follower-zero.json')
    parser.add_argument('--obstacle-xy', type=float, nargs=2, required=True,
                        metavar=('X', 'Y'))
    parser.add_argument('--home-tolerance-deg', type=float, default=5.0)
    parser.add_argument('--margin', type=float, default=0.002)
    parser.add_argument('--delay', type=float, default=0.2,
                        help='Seconds between displayed trajectory points')
    parser.add_argument('--motion-config', type=Path,
                        default=PROJECT_ROOT / 'scripts/hardware/follower-home.json')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if (args.joints is not None and not np.isfinite(args.joints).all()
            or not np.isfinite(args.obstacle_xy).all()
            or not np.isfinite([args.home_tolerance_deg, args.margin, args.delay]).all()
            or min(args.home_tolerance_deg, args.margin, args.delay) < 0):
        parser.error('Joint values, obstacle position, margin and delay must be finite and nonnegative where applicable')

    config = json.loads(args.motion_config.read_text(encoding='utf-8'))
    home_data = json.loads((PROJECT_ROOT / 'scripts/so101/training_home.json').read_text(encoding='utf-8'))
    home = dict(zip(home_data['joint_names'], home_data['positions_rad'], strict=True))
    if args.read_current:
        if args.calibration is None:
            parser.error('--calibration is required with --read-current')
        joints = read_current_joints(
            args.port, args.calibration, args.mapping, config, home,
        )
    else:
        joints = np.asarray(args.joints, dtype=float)
    env = RelativePlanningEnv(curriculum_stage=1)
    try:
        env._place(env.start_pose)
        configure_rollout(env, config['joints'], JOINT_NAMES)
        policy = PPO.load(args.checkpoint, env=env, device='cpu')
        obstacle_xy = np.asarray(args.obstacle_xy, dtype=float)
        target, trajectory = find_short_candidate(
            env, policy, joints, obstacle_xy, args.margin,
            np.deg2rad(args.home_tolerance_deg),
        )
        env.obstacle_center_z = 0.13
        print(f'Replaying rejected simulated candidate with {len(trajectory) - 1} steps.')
        print(f'Target (m): {target.round(3).tolist()}')
        print(f'Box center (m): {obstacle_xy.round(3).tolist()}, 0.130')
        preview(env, target, trajectory, obstacle_xy, args.delay)
    finally:
        env.close()


if __name__ == '__main__':
    main()
