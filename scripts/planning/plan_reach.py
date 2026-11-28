"""Plan an outbound reach and optionally validate a return in MuJoCo."""

import argparse
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import PPO

from scripts import PROJECT_ROOT
from scripts.simulation.static_obstacle_env import StaticObstacleEnv

DEFAULT_CHECKPOINT = (PROJECT_ROOT /
                      "outputs/static-curriculum-stage1-continued/best/best_model.zip")


class PlanningEnv(StaticObstacleEnv):
    """Measure arm-to-block clearance at the same physics states as collision checks."""

    def _status(self):
        collision, unsafe = super()._status()
        if not unsafe:
            for geom in self.arm_geoms:
                gap = mujoco.mj_geomDistance(self.model, self.data, self.block_geom,
                                             int(geom), self.min_gap, None)
                if not np.isfinite(gap):
                    return collision, True
                self.min_gap = min(self.min_gap, gap)
                if self.min_gap <= 0:
                    return True, unsafe
        return collision, unsafe


class RelativePlanningEnv(PlanningEnv):
    """Test the follower's relative moves against the MuJoCo collision geometry."""

    def _advance_physics(self):
        target = np.clip(self.data.ctrl[self.actuators],
                         *self.model.actuator_ctrlrange[self.actuators].T)
        self.data.ctrl[self.actuators] = target
        self.data.qpos[self.addresses] = target
        self.data.time += self.model.opt.timestep


def candidate_targets(env, start, obstacle, direction, margin, seed):
    """Use sampled joint poses as reachability witnesses, as in training."""
    low, high = env.limits[:5].T
    rng = np.random.default_rng(seed)
    poses = rng.uniform(np.maximum(low + 0.05, env.start_pose - 0.75),
                        np.minimum(high - 0.05, env.start_pose + 0.75), (24000, 5))
    buckets = {}
    half_size = env.obstacle_half_size[:2]
    far_edge = np.abs(direction) @ half_size
    for pose in poses:
        env.data.qpos[env.addresses[:5]] = pose
        mujoco.mj_kinematics(env.model, env.data)
        tip = env.data.site_xpos[env.tip]
        if abs(tip[2] - env.target_height) > 0.005:
            continue
        target = np.array([tip[0], tip[1], env.target_height])
        distance = np.linalg.norm(target - start)
        if not env.min_target_distance <= distance <= env.max_target_distance:
            continue
        if not (0.12 < tip[0] < 0.38 and -0.25 < tip[1] < 0.15):
            continue
        offset = target[:2] - obstacle
        gap = np.linalg.norm(np.maximum(np.abs(offset) - half_size, 0))
        if not max(margin, env.tolerance) <= gap <= 0.06:
            continue
        far = bool(offset @ direction > far_edge)
        angle = np.arctan2(offset[1], offset[0])
        sector = int(np.floor((angle % (2 * np.pi)) / (np.pi / 6)))
        key = (far, sector)
        if key in buckets:
            continue
        mujoco.mj_forward(env.model, env.data)
        if not env._collision():
            buckets[key] = target
    return [(far, target) for (far, _), target in buckets.items()]


def plan(env, policy, joints, obstacle_xy, *, margin, home_tolerance=0.0, seed=0,
         prepare_candidate=None, far_only=False):
    """Return a simulated trajectory and rejection counts. A failed plan returns None."""
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Clearance margin must be finite and nonnegative.")
    obstacle = np.asarray(obstacle_xy, dtype=float)
    rejected = Counter()
    try:
        start = env.place_planning_start(joints, obstacle, home_tolerance)
    except ValueError as exc:
        return None, Counter({str(exc): 1})
    direction = obstacle - start[:2]
    length = np.linalg.norm(direction)
    if length == 0:
        return None, Counter({"undefined approach direction": 1})
    direction /= length
    targets = candidate_targets(env, start, obstacle, direction, margin, seed)
    if not targets:
        return None, Counter({"no reachable candidates sampled": 1})
    far_edge = np.abs(direction) @ env.obstacle_half_size[:2]
    for prefer_far in (True, False):
        if far_only and not prefer_far:
            break
        best = None
        for far, target in targets:
            if far != prefer_far:
                continue
            try:
                obs, _ = env.reset_for_planning(joint_positions=joints, obstacle_xy=obstacle,
                                                target_position=target, home_tolerance=home_tolerance)
            except ValueError as exc:
                rejected[str(exc)] += 1
                continue
            # The entire scene is smaller than two meters.
            env.min_gap = 2.0
            points = [trajectory_point(env)]
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, done, timeout, info = env.step(action)
                points.append(trajectory_point(env))
                if info["collision"]:
                    rejected["collision"] += 1
                elif info["unsafe"]:
                    rejected["unsafe state"] += 1
                elif env.min_gap < margin:
                    rejected["insufficient clearance"] += 1
                elif timeout:
                    rejected["timeout"] += 1
                elif done:
                    end_offset = env.data.site_xpos[env.tip, :2] - obstacle
                    if far and end_offset @ direction <= far_edge:
                        rejected["gripper stopped short of far side"] += 1
                    elif info["is_success"]:
                        rank = (env.min_gap, -env.data.time)
                        if best is None or rank > best["rank"]:
                            candidate = dict(rank=rank, target=target.copy(), far_side=far,
                                        obstacle_xy=obstacle.copy(), margin=margin,
                                        end_velocity=env.data.qvel.copy(),
                                        min_clearance=env.min_gap, trajectory=points)
                            if prepare_candidate is not None:
                                try:
                                    candidate['trajectory'] = prepare_candidate(env, candidate)
                                except ValueError as exc:
                                    rejected[str(exc)] += 1
                                    break
                            best = candidate
                else:
                    continue
                break
        if best is not None:
            return best, rejected
    return None, rejected


def trajectory_point(env):
    return dict(time=env.data.time, positions=env.data.qpos[env.addresses].copy(),
                commands=env.data.ctrl[env.actuators].copy(),
                tip=env.data.site_xpos[env.tip].copy())


def validate_return(env, outbound, measured_positions, measured_velocities, obstacle_xy,
                    *, home_tolerance=0.01, stationary_speed=0.01):
    """Replay reversed joint positions from measured state and verify settled home.

    The current model has six hinge joints. Positions are in training order,
    velocities in MuJoCo DOF order. Tolerances apply to every joint.
    """
    if outbound is None:
        return None, "missing outbound trajectory"
    obstacle = np.asarray(obstacle_xy, dtype=float)
    if not np.array_equal(obstacle, outbound["obstacle_xy"]):
        return None, "obstacle position changed"
    joints = np.asarray(measured_positions, dtype=float).copy()
    velocity = np.asarray(measured_velocities, dtype=float).copy()
    if (joints.shape != (len(env.addresses),) or velocity.shape != (env.model.nv,)
            or not np.isfinite(joints).all() or not np.isfinite(velocity).all()):
        return None, "invalid joint measurement"
    if not all(np.isfinite(x) and x >= 0 for x in (home_tolerance, stationary_speed)):
        raise ValueError("Home and stationary tolerances must be finite and nonnegative.")

    mujoco.mj_resetData(env.model, env.data)
    env._place(env.start_pose)
    home = env.data.qpos[env.addresses].copy()
    env.data.qpos[env.addresses] = joints
    env.data.qvel[:] = velocity
    low, high = env.limits.T
    env.data.ctrl[env.actuators] = np.clip(joints, low, high)
    env._set_block(obstacle)
    env.min_gap = 2.0

    def rejection():
        collision, unsafe = env._status()
        if collision:
            return "collision"
        if unsafe:
            return "unsafe joint state or speed"
        if env.min_gap < outbound["margin"]:
            return "insufficient clearance"
        return None

    def advance():
        before = env.data.time
        mujoco.mj_step(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
        if env.data.time <= before:
            return "simulation time failed to advance"
        return rejection()

    reason = rejection()
    if reason:
        return None, reason
    trajectory = [trajectory_point(env)]
    waypoints = [point["positions"] for point in reversed(outbound["trajectory"])]
    waypoints.append(home)
    dt = env.model.opt.timestep
    # Match the outbound command-rate cap. Check dynamics at every physics step.
    command_speed = min(0.5, 0.02 / env.control_dt)
    previous = env.data.ctrl[env.actuators].copy()
    for sample in waypoints:
        # MuJoCo's soft stops allow small measured excursions. Use the same
        # 0.001 rad state tolerance as ReachingEnv._status and bound commands.
        if np.any(sample < low - 0.001) or np.any(sample > high + 0.001):
            return None, "return waypoint outside joint limits"
        goal = np.clip(sample, low, high)
        delta = goal - previous
        if not np.any(delta):
            continue
        steps = max(env.substeps, int(np.ceil(np.max(np.abs(delta)) / (command_speed * dt))))
        for i in range(1, steps + 1):
            env.data.ctrl[env.actuators] = previous + delta * (i / steps)
            reason = advance()
            if reason:
                return None, reason
        trajectory.append(trajectory_point(env))
        previous = goal

    # Hold home for up to one second. Require a continuous 0.1 s settled interval.
    env.data.ctrl[env.actuators] = home
    settled_steps = 0
    for _ in range(int(np.ceil(1.0 / dt))):
        reason = advance()
        if reason:
            return None, reason
        error = np.max(np.abs(env.data.qpos[env.addresses] - home))
        speed = np.max(np.abs(env.data.qvel))
        settled_steps = settled_steps + 1 if error <= home_tolerance and speed <= stationary_speed else 0
        if settled_steps * dt >= 0.1:
            trajectory.append(trajectory_point(env))
            return dict(trajectory=trajectory, min_clearance=env.min_gap,
                        home_error=error, final_speed=speed), None
    return None, "did not settle at training home"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obstacle-xy", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--min-clearance", type=float, required=True, help="Simulation margin in meters")
    parser.add_argument("--validate-return", action="store_true",
                        help="Independently simulate the return after an accepted reach")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    env = PlanningEnv(curriculum_stage=1)
    try:
        model = PPO.load(args.checkpoint, env=env, device="cpu")
        env._place(env.start_pose)
        result, rejected = plan(env, model, env.data.qpos[env.addresses].copy(),
                                args.obstacle_xy, margin=args.min_clearance)
        if rejected:
            print("Rejected:", dict(rejected))
        if result is None:
            print("no_valid_trajectory")
            return 1
        print("Target (m):", result["target"], "Far side:", result["far_side"])
        print("Minimum clearance (m):", result["min_clearance"])
        print("Simulated duration (s):", result["trajectory"][-1]["time"])
        if args.validate_return:
            # Use the selected rollout's simulated feedback. The search may
            # have left env.data at the end of a different candidate.
            returned, reason = validate_return(env, result, result["trajectory"][-1]["positions"],
                                               result["end_velocity"], args.obstacle_xy)
            if returned is None:
                print("no_valid_return:", reason)
                return 1
            print("Return duration (s):", returned["trajectory"][-1]["time"])
            print("Return minimum clearance (m):", returned["min_clearance"])
            print("Maximum home error (rad):", returned["home_error"])
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
