"""Evaluate saved PPO policies in simulation on identical seeded scenarios."""

import argparse
import csv
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import PPO
import torch

from scripts.simulation.reaching_env import ARM
from scripts.simulation.static_obstacle_env import CURRICULUM, TASK_ENVS


def evaluate(
    checkpoint, episodes, seed, render, trace_dir=None,
    task="reaching", curriculum_stage=0,
):
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=False)
    env_class = TASK_ENVS[task]
    kwargs = (
        {"curriculum_stage": curriculum_stage}
        if task == "static-obstacle" else {}
    )
    env = env_class(render_mode="human" if render else None, **kwargs)
    results = []
    sides = []
    try:
        # Prediction needs no attached training environment; stage bounds may
        # differ while retaining the same 19 inputs and five outputs.
        model = PPO.load(checkpoint, device="cpu")
        if (model.observation_space.shape != env.observation_space.shape
                or model.action_space != env.action_space):
            raise ValueError("Checkpoint and evaluation environment are incompatible.")
        print(f"\nCheckpoint: {checkpoint}", flush=True)
        print(f"Task: {task}; curriculum stage: {curriculum_stage}", flush=True)
        print(f"Training steps: {model.num_timesteps}", flush=True)
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            initial_distance = closest = info["distance"]
            # Forward follows the arm's initial horizontal pointing direction.
            start_xy = env.data.site_xpos[env.tip, :2].copy()
            shoulder_xy = env.data.xanchor[
                env.model.joint("shoulder_pan").id, :2
            ]
            forward = start_xy - shoulder_xy
            forward /= np.linalg.norm(forward)
            right = np.array([forward[1], -forward[0]])
            lateral = float(np.dot(env.goal[:2] - start_xy, right))
            side = "right" if lateral > 0.01 else (
                "left" if lateral < -0.01 else "center"
            )
            sides.append(side)
            total_reward = 0.0
            trace = []
            for step in range(env.max_steps):
                started = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                if trace_dir is not None:
                    before = env.data.qpos[env.addresses[:5]].copy()
                    distance_before = info["distance"]
                    time_before = float(env.data.time)
                obs, reward, terminated, truncated, info = env.step(action)
                if trace_dir is not None:
                    row = {
                        "step": step + 1,
                        "target_side": side,
                        "target_lateral_m": lateral,
                        "time_before_s": time_before,
                        "time_after_s": float(env.data.time),
                        "distance_before_m": distance_before,
                        "distance_after_m": info["distance"],
                        "reward": reward,
                        "success": info["is_success"],
                        "collision": info["collision"],
                        "unsafe": info["unsafe"],
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                    for i, joint in enumerate(ARM):
                        row[f"{joint}_action"] = float(action[i])
                        row[f"{joint}_before_rad"] = float(before[i])
                        row[f"{joint}_after_rad"] = float(
                            env.data.qpos[env.addresses[i]]
                        )
                        # Last servo setpoint applied during this step.
                        row[f"{joint}_command_rad"] = float(
                            env.data.ctrl[env.actuators[i]]
                        )
                    for i, axis in enumerate("xyz"):
                        row[f"target_{axis}_m"] = float(env.goal[i])
                        row[f"tip_{axis}_m"] = float(
                            env.data.site_xpos[env.tip, i]
                        )
                    trace.append(row)
                total_reward += reward
                closest = min(closest, info["distance"])
                if render:
                    if not env._viewer.is_running():
                        print("Viewer closed; evaluation stopped.")
                        return
                    time.sleep(max(
                        0.0, env.control_dt - (time.perf_counter() - started)
                    ))
                if terminated or truncated:
                    break

            if trace:
                path = trace_dir / f"seed-{seed + episode}.csv"
                with path.open("x", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=list(trace[0]))
                    writer.writeheader()
                    writer.writerows(trace)
                print(f"Trace: {path}", flush=True)

            results.append([
                info["is_success"], info["collision"], info["unsafe"],
                truncated, total_reward, step + 1,
                initial_distance, info["distance"], closest,
            ])
            outcomes = [
                name for name, occurred in (
                    ("success", info["is_success"]),
                    ("collision", info["collision"]),
                    ("unsafe", info["unsafe"]),
                    ("timeout", truncated),
                ) if occurred
            ]
            print(
                f"Episode {episode + 1}/{episodes} "
                f"seed={seed + episode}, robot-side={side}: {'/'.join(outcomes)}, "
                f"steps={step + 1}, reward={total_reward:.2f}, "
                f"distance={initial_distance * 100:.1f}"
                f" -> {info['distance'] * 100:.1f} cm, "
                f"closest={closest * 100:.1f} cm",
                flush=True,
            )

        results = np.asarray(results)
        means = np.mean(results, axis=0)
        print(f"Success: {means[0]:.1%}")
        print(f"Collision: {means[1]:.1%}; unsafe: {means[2]:.1%}")
        print(f"Timeout: {means[3]:.1%}")
        first_action_collision = np.mean(
            (results[:, 1] == 1) & (results[:, 5] == 1)
        )
        print(f"First-action collision: {first_action_collision:.1%}")
        print(f"Mean reward: {means[4]:.2f}; mean steps: {means[5]:.1f}")
        print(f"Mean distance: {means[6] * 100:.1f}"
              f" -> {means[7] * 100:.1f} cm")
        print(f"Mean closest distance: {means[8] * 100:.1f} cm")
        groups = []
        sides = np.asarray(sides)
        for side in ("left", "center", "right"):
            mask = sides == side
            groups.append((f"Robot {side}", mask))
        for label, mask in groups:
            group = results[mask]
            if len(group):
                print(
                    f"{label}: n={len(group)}, success={group[:, 0].mean():.1%}, "
                    f"mean steps={group[:, 5].mean():.1f}, "
                    f"mean final distance={group[:, 7].mean() * 100:.1f} cm"
                )
            else:
                print(f"{label}: n=0")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--task", choices=TASK_ENVS, default="reaching",
        help="Environment used to evaluate the checkpoint.",
    )
    parser.add_argument(
        "--trace-dir", type=Path,
        help="Write per-step CSV traces into new checkpoint subdirectories.",
    )
    parser.add_argument(
        "--curriculum-stage", type=int, choices=(0, *CURRICULUM), default=0,
        help="Static obstacle stage: 0 uses the original sampler; 1 uses curriculum.",
    )
    args = parser.parse_args()
    if args.curriculum_stage and args.task != "static-obstacle":
        parser.error("--curriculum-stage requires --task static-obstacle.")
    if args.episodes < 1 or args.seed < 0:
        parser.error("Episodes must be positive and seed nonnegative.")
    torch.set_num_threads(1)
    for index, checkpoint in enumerate(args.checkpoints):
        trace_dir = (
            args.trace_dir / f"{index:02d}-{checkpoint.stem}"
            if args.trace_dir is not None else None
        )
        evaluate(
            checkpoint, args.episodes, args.seed, args.render, trace_dir,
            task=args.task, curriculum_stage=args.curriculum_stage,
        )


if __name__ == "__main__":
    main()
