"""Home the follower, plan around a measured box, and replay relative joint moves."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.hiwonder import HiwonderMotorsBus, MODEL_RESOLUTION

from scripts import PROJECT_ROOT
from scripts.planning.plan_reach import DEFAULT_CHECKPOINT, PPO, RelativePlanningEnv, plan
from scripts.simulation.reaching_env import JOINT_NAMES
from scripts.so101.calibration import connected_reader, load_calibration, load_mapping
from scripts.so101.geometry import check_pose, check_segment, configure_rollout
from scripts.so101.hiwonder_control import HiwonderArm


def measured(arm, *, settle=False):
    deadline = time.monotonic() + 3.0
    while True:
        pose, _, moving = arm.sample(enforce_speed=False)
        if not settle or not moving:
            return np.array(pose)
        if time.monotonic() >= deadline:
            raise RuntimeError('Arm still moving after three seconds')
        time.sleep(0.05)


def send_move(arm, env, start, target, margin):
    ticks = arm.ticks(target.tolist())
    rounded = np.array(arm.tick_positions(ticks))
    check_pose(env, start, margin)
    check_segment(env, start, rounded, margin)
    arm.send(ticks)


def run_reach(arm, env, policy, home, tolerance, margin):
    speeds = np.array([arm.limits[name][2] for name in arm.names])
    if input('Clear the workspace, then type home: ').strip() != 'home':
        return
    start = measured(arm, settle=True)
    offset = home - start
    check_pose(env, start, margin)
    check_segment(env, start, home, margin)
    arm.engage(start.tolist(), enforce_speed=False)
    # One straight relative move, paced to the slowest participating joint.
    frames = max(1, int(np.ceil(np.max(np.abs(offset) / speeds) / env.control_dt)))
    for frame in range(1, frames + 1):
        send_move(arm, env, measured(arm), start + offset * (frame / frames), margin)
        time.sleep(env.control_dt)
    if np.max(np.abs(measured(arm, settle=True) - home)) > tolerance:
        raise RuntimeError('Arm finished outside the home tolerance')

    center = [float(v) for v in input('Place the box. Enter center X Y Z in world meters: ').split()]
    if len(center) != 3 or not np.isfinite(center).all():
        raise ValueError('Expected three finite coordinates')
    xy, env.obstacle_center_z = np.array(center[:2]), center[2]
    start = measured(arm, settle=True)
    result, rejected = plan(env, policy, start, xy, margin=margin, home_tolerance=tolerance,
                            far_only=True)
    if result is None:
        raise RuntimeError(f'No valid far-side reach: {dict(rejected)}')

    points = result['trajectory']
    print(f"Target {result['target'].round(3)} m, {len(points)-1} policy steps, "
          f"simulated clearance {result['min_clearance']*1000:.1f} mm")
    if input('Type go to replay: ').strip() != 'go':
        return
    if np.max(np.abs(measured(arm, settle=True) - start)) > tolerance:
        raise RuntimeError('Arm moved since planning')
    bounds = env.model.actuator_ctrlrange[env.actuators[:5]].T
    for before, after in zip(points, points[1:]):
        # PPO commands each increment from the joint measurement at the start of its step.
        delta = after['commands'][:5] - before['positions'][:5]
        current = measured(arm)
        target = np.r_[np.clip(current[:5] + delta, *bounds), home[-1]]
        send_move(arm, env, current, target, margin)
        time.sleep(max(env.control_dt, np.max(np.abs(target - current) / speeds)))

    check_pose(env, measured(arm, settle=True), margin)
    tip = env.data.site_xpos[env.tip]
    direction = xy - points[0]['tip'][:2]
    beyond = (tip[:2] - xy) @ direction > env.obstacle_half_size[:2] @ np.abs(direction)
    error = np.linalg.norm(tip - result['target'])
    if not beyond or error > env.tolerance:
        raise RuntimeError(f'Reach missed: tip error {error*100:.1f} cm, beyond box: {beyond}')
    print(f'Reached the far side. Tip error {error*100:.1f} cm.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyACM0')
    parser.add_argument('--calibration', type=Path, default=os.getenv('FOLLOWER_CALIBRATION'),
                        required=not os.getenv('FOLLOWER_CALIBRATION'))
    parser.add_argument('--mapping', type=Path, default=PROJECT_ROOT / 'outputs/follower-zero.json')
    parser.add_argument('--motion-config', type=Path, default=PROJECT_ROOT / 'outputs/follower-home.json')
    parser.add_argument('--home-tolerance-deg', type=float, default=3.0)
    args = parser.parse_args()
    limits = json.loads(args.motion_config.read_text())
    margin = float(limits.get('policy_clearance_m', 0.002))
    tolerance = np.deg2rad(args.home_tolerance_deg)
    if not np.isfinite([tolerance, margin]).all() or min(tolerance, margin) < 0:
        parser.error('Home tolerance and clearance must be finite and nonnegative')

    with closing(RelativePlanningEnv(curriculum_stage=1)) as env:
        env._place(env.start_pose)
        home = env.data.qpos[env.addresses].copy()
        configure_rollout(env, limits['joints'], JOINT_NAMES)
        policy = PPO.load(DEFAULT_CHECKPOINT, env=env, device='cpu')
        max_tick = MODEL_RESOLUTION['hx30hm'] - 1
        rows, digest = load_calibration(args.calibration, JOINT_NAMES, max_tick)
        calibration = {name: rows[name] for name in JOINT_NAMES}
        mapping = load_mapping(args.mapping, digest, calibration, dict(zip(JOINT_NAMES, home)),
                               2 * np.pi / max_tick)
        motors = {n: Motor(row['id'], 'hx30hm', MotorNormMode.RANGE_0_100
                           if n == 'gripper' else MotorNormMode.DEGREES)
                  for n, row in calibration.items()}
        bus = HiwonderMotorsBus(port=args.port, motors=motors, calibration={
            n: MotorCalibration(**row) for n, row in calibration.items()})
        arm = HiwonderArm(bus, calibration, mapping, limits)
        with connected_reader(bus, calibration, exclusive=True):
            try:
                run_reach(arm, env, policy, home, tolerance, margin)
            finally:
                arm.stop()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (EOFError, OSError, RuntimeError, ValueError) as exc:
        sys.exit(f'arm_reach: {exc}')
