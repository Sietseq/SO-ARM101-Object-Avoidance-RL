"""Train a simulation-only reaching policy with SB3 PPO."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from static_obstacle_env import CURRICULUM, TASK_ENVS


class SuccessEvalCallback(EvalCallback):
    """Evaluate a fixed seeded set; save by success rate, then mean reward."""

    def __init__(self, eval_env, *, best_model_save_path, eval_seed, **kwargs):
        super().__init__(eval_env, best_model_save_path=None, verbose=0, **kwargs)
        self.success_path = Path(best_model_save_path)
        self.eval_seed = eval_seed
        self.best_score = (float("-inf"), float("-inf"))

    def _on_training_start(self):
        # Establish and save the starting policy before PPO changes it.
        self._on_step()

    def _on_step(self):
        due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if due:
            self.eval_env.seed(self.eval_seed)
        keep_training = super()._on_step()
        if due:
            if not self._is_success_buffer:
                raise RuntimeError("Evaluation must report is_success.")
            score = (float(np.mean(self._is_success_buffer)), self.last_mean_reward)
            print(f"Evaluation: success={score[0]:.1%}, mean reward={score[1]:.2f}",
                  flush=True)
            if score > self.best_score:
                self.success_path.mkdir(parents=True, exist_ok=True)
                self.model.save(self.success_path / "best_model")
                self.best_score = score
                print("Saved best policy by success rate (reward breaks ties).",
                      flush=True)
        return keep_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timesteps", type=int, default=300_000,
        help="Training steps to run, additional to any resumed checkpoint.",
    )
    parser.add_argument(
        "--resume", type=Path,
        help="Continue training from a PPO checkpoint (.zip).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument(
        "--action-std", type=float, default=None,
        help="Initial Gaussian exploration standard deviation in action units.",
    )
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", choices=TASK_ENVS, default="reaching",
        help="Train plain reaching or randomized static-block avoidance.",
    )
    parser.add_argument(
        "--curriculum-stage", type=int, choices=(0, *CURRICULUM), default=0,
        help="0: original sampler; 1: static-obstacle curriculum.",
    )
    parser.add_argument(
        "--clear-episode-prob", type=float, default=0.0,
        help="Obstacle-free fraction of training episodes only.",
    )
    args = parser.parse_args()
    if not 0 <= args.clear_episode_prob <= 1:
        parser.error("--clear-episode-prob must be in [0, 1].")
    if args.task != "static-obstacle" and (
        args.curriculum_stage or args.clear_episode_prob
    ):
        parser.error("Curriculum options require --task static-obstacle.")
    if args.resume is not None and not args.resume.is_file():
        parser.error("--resume must point to an existing checkpoint file.")
    if min(args.timesteps, args.eval_freq, args.eval_episodes) < 1:
        parser.error("Timesteps and evaluation settings must be positive.")
    if args.seed < 0:
        parser.error("Seed must be nonnegative.")
    if args.n_steps < 2 or args.batch_size < 2:
        parser.error("Rollout and batch sizes must be at least two.")
    if args.n_steps % args.batch_size:
        parser.error("Batch size must divide the single-environment rollout.")

    for name in ("learning_rate", "target_kl", "action_std"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name.replace('_', '-')} must be finite and positive.")

    # Require a new directory so previous experiments remain intact.
    args.output.mkdir(parents=True, exist_ok=False)
    env_class = TASK_ENVS[args.task]
    env_kwargs = (
        {"curriculum_stage": args.curriculum_stage}
        if args.task == "static-obstacle" else {}
    )
    train_kwargs = dict(env_kwargs)
    if args.task == "static-obstacle":
        train_kwargs["clear_episode_prob"] = args.clear_episode_prob
    train_env = env_class(**train_kwargs)
    eval_env = None
    try:
        check_env(train_env, warn=True)
        train_env = Monitor(
            train_env, str(args.output / "train.monitor.csv"),
            info_keywords=("is_success", "collision", "unsafe", "distance"),
        )
        # Select checkpoints on obstacle-present layouts from the current stage.
        eval_env = Monitor(env_class(**env_kwargs))
        eval_env.reset(seed=args.seed + 1)
        callback = SuccessEvalCallback(
            eval_env,
            best_model_save_path=str(args.output / "best"),
            eval_seed=args.seed + 1,
            log_path=str(args.output / "evaluation"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )
        if args.resume is not None:
            overrides = {
                name: getattr(args, name) for name in ("learning_rate", "target_kl")
                if getattr(args, name) is not None
            }
            load_options = {}
            if args.task != "reaching":
                # Obstacle variants reuse the 19 inputs with task-specific bounds.
                load_options["custom_objects"] = {
                    "observation_space": train_env.observation_space
                }
            model = PPO.load(
                args.resume, env=train_env, device=args.device,
                seed=args.seed, n_steps=args.n_steps,
                batch_size=args.batch_size, **load_options, **overrides,
            )
            print(f"Continuing from {model.num_timesteps} trained steps.")
        else:
            model = PPO(
                "MlpPolicy", train_env,
                learning_rate=(
                    args.learning_rate
                    if args.learning_rate is not None else 3e-4
                ),
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                target_kl=args.target_kl,
                seed=args.seed,
                device=args.device,
                verbose=1,
            )
        if args.action_std is not None:
            if not isinstance(model.policy.action_dist, DiagGaussianDistribution):
                raise ValueError("--action-std requires a diagonal Gaussian policy.")
            with torch.no_grad():
                model.policy.log_std.fill_(math.log(args.action_std))
        model.set_logger(configure(str(args.output), ["stdout", "csv"]))
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=args.resume is None,
            )
        except KeyboardInterrupt:
            model.save(args.output / "interrupted_model")
            raise
        model.save(args.output / "final_model")
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
