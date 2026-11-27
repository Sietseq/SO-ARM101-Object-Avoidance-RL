"""Train a simulation-only reaching or static-obstacle policy with SB3 PPO."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from reaching_env import ReachingEnv
from static_obstacle_env import StaticObstacleEnv


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
    parser.add_argument(
        "--task", choices=("reaching", "static-obstacle"), default="reaching",
    )
    parser.add_argument(
        "--blocked-fraction", type=float, default=0.2,
        help="Fraction of static-obstacle training targets requiring a detour.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.resume is not None and not args.resume.is_file():
        parser.error("--resume must point to an existing checkpoint file.")
    if min(args.timesteps, args.eval_freq, args.eval_episodes) < 1:
        parser.error("Timesteps and evaluation settings must be positive.")
    if not 0.0 <= args.blocked_fraction <= 1.0:
        parser.error("Blocked fraction must be between zero and one.")
    if args.seed < 0:
        parser.error("Seed must be nonnegative.")
    if args.n_steps < 2 or args.batch_size < 2:
        parser.error("Rollout and batch sizes must be at least two.")
    if args.n_steps % args.batch_size:
        parser.error("Batch size must divide the single-environment rollout.")

    # Require a new directory so previous experiments remain intact.
    args.output.mkdir(parents=True, exist_ok=False)
    env_class = StaticObstacleEnv if args.task == "static-obstacle" else ReachingEnv
    train_kwargs = (
        {"pool_seed": args.seed, "blocked_fraction": args.blocked_fraction}
        if args.task == "static-obstacle" else {}
    )
    train_env = env_class(**train_kwargs)
    eval_env = None
    try:
        check_env(train_env, warn=True)
        train_env = Monitor(
            train_env, str(args.output / "train.monitor.csv"),
            info_keywords=("is_success", "collision", "unsafe", "distance"),
        )
        eval_kwargs = (
            {"pool_size": 128, "pool_seed": args.seed + 1, "blocked_fraction": 1.0}
            if args.task == "static-obstacle" else {}
        )
        eval_env = Monitor(env_class(**eval_kwargs))
        eval_env.reset(seed=args.seed + 1)
        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(args.output / "best"),
            log_path=str(args.output / "evaluation"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )
        if args.resume is not None:
            model = PPO.load(
                args.resume, env=train_env, device=args.device,
                seed=args.seed, n_steps=args.n_steps,
                batch_size=args.batch_size,
            )
            print(f"Continuing from {model.num_timesteps} trained steps.")
        else:
            model = PPO(
                "MlpPolicy", train_env,
                learning_rate=3e-4,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                seed=args.seed,
                device=args.device,
                verbose=1,
            )
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
