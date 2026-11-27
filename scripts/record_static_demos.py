"""Record successful simulated expert rollouts for static-obstacle imitation."""

import argparse
from pathlib import Path

import numpy as np

from static_obstacle_env import StaticObstacleEnv


def collect_demonstrations(episodes, seed=0, blocked_fraction=0.8):
    env = StaticObstacleEnv(pool_size=0, blocked_fraction=blocked_fraction)
    observations, actions, episode_ids, seeds, blocked_episodes = [], [], [], [], []
    canonical_waypoint = None
    try:
        for attempt in range(episodes * 10):
            episode_seed = seed + attempt
            env.blocked_fraction = float(len(seeds) < round(episodes * blocked_fraction))
            env.reset(seed=episode_seed)
            start = env.data.qpos[env.addresses[:5]].copy()
            blocked = env._blocks_segment(env.data.site_xpos[env.tip], env.goal)
            if blocked and canonical_waypoint is None:
                canonical_waypoint = env._waypoint_pose.copy()
            waypoint = canonical_waypoint if blocked else env._waypoint_pose
            goal_pose = env._goal_pose.copy()
            if not (env._clear_segment(start, waypoint)
                    and env._clear_segment(waypoint, goal_pose)):
                continue
            env._place(start)
            obs = env._observation()
            trajectory = []
            approaching_goal = False
            for _ in range(env.max_steps):
                q = env.data.qpos[env.addresses[:5]]
                if np.max(np.abs(waypoint - q)) < 0.01:
                    approaching_goal = True
                error = (goal_pose if approaching_goal else waypoint) - q
                # Coordinate joint speeds to follow the checked joint segment.
                increment = min(0.02, 0.5 * env.control_dt)
                action = (error / max(increment, float(np.max(np.abs(error)))))
                action = action.astype(np.float32)
                trajectory.append((obs.copy(), action.copy()))
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            if info["is_success"] and not (info["collision"] or info["unsafe"]):
                episode_id = len(seeds)
                for observation, action in trajectory:
                    observations.append(observation)
                    actions.append(action)
                    episode_ids.append(episode_id)
                seeds.append(episode_seed)
                blocked_episodes.append(blocked)
            if len(seeds) == episodes or (attempt + 1) % 25 == 0:
                print(f"Kept {len(seeds)}/{episodes} successful demonstrations "
                      f"after {attempt + 1} attempts.", flush=True)
            if len(seeds) == episodes:
                break
        else:
            raise RuntimeError("Too few successful demonstrations; no dataset saved.")
    finally:
        env.close()
    return {
        "format_version": np.array(1),
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "blocked": np.asarray(blocked_episodes, dtype=bool),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blocked-fraction", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 2 or args.seed < 0:
        parser.error("Use at least two episodes and a nonnegative seed.")
    if not 0 <= args.blocked_fraction <= 1:
        parser.error("Blocked fraction must be between zero and one.")
    if args.output.exists():
        parser.error("Output must be a new dataset file.")
    data = collect_demonstrations(args.episodes, args.seed, args.blocked_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        np.savez_compressed(output, **data)
    print(f"Saved {len(data['actions'])} observation/action pairs to {args.output}.")


if __name__ == "__main__":
    main()
