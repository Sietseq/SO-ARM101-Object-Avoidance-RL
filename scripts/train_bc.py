"""Warm-start an existing PPO actor with behavior cloning (MSE and Adam)."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution

from static_obstacle_env import StaticObstacleEnv


def validate_data(data):
    obs, actions, ids = (data[k] for k in ("observations", "actions", "episode_ids"))
    if int(data["format_version"]) != 1:
        raise ValueError("Unsupported demonstration format.")
    if obs.ndim != 2 or obs.shape[1] != 19 or actions.shape != (len(obs), 5):
        raise ValueError("Expected observations[N,19] and actions[N,5].")
    if ids.shape != (len(obs),) or ids.dtype.kind not in "iu":
        raise ValueError("Expected one integer episode ID per transition.")
    if (len(np.unique(ids)) < 2 or not np.isfinite(obs).all()
            or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)):
        raise ValueError("Need finite, bounded data from at least two episodes.")
    if (data["seeds"].ndim != 1 or len(data["seeds"]) < 2
            or np.any(ids < 0) or np.any(ids >= len(data["seeds"]))):
        raise ValueError("Invalid episode seed mapping.")


def evaluate(model, env, episodes, seed):
    successes = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        successes.append(info["is_success"])
    return float(np.mean(successes))


def clone_actor(model, data, *, epochs=100, batch_size=256, learning_rate=3e-4,
                seed=0, eval_episodes=20, eval_every=10):
    validate_data(data)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Split whole target episodes; adjacent transitions never cross the split.
    episodes = rng.permutation(np.unique(data["episode_ids"]))
    validation_ids = episodes[:max(1, round(len(episodes) * 0.2))]
    validation = np.isin(data["episode_ids"], validation_ids)
    obs = torch.as_tensor(data["observations"], dtype=torch.float32, device=model.device)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=model.device)
    train_ids = torch.as_tensor(np.flatnonzero(~validation), device=model.device)
    val_ids = torch.as_tensor(np.flatnonzero(validation), device=model.device)
    policy = model.policy
    if not isinstance(policy.action_dist, DiagGaussianDistribution):
        raise ValueError("Behavior cloning requires the project's Gaussian MLP policy.")
    actor_parameters = (
        list(policy.pi_features_extractor.parameters())
        + list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
    )
    optimizer = torch.optim.Adam(actor_parameters, lr=learning_rate)
    # Independent blocked layouts assess closed-loop avoidance after cloning.
    eval_seed = int(np.max(data["seeds"])) + 1
    eval_env = StaticObstacleEnv(pool_size=128, pool_seed=eval_seed,
                                 blocked_fraction=1.0)
    history, best_state, best_score = [], None, (-1.0, float("-inf"))
    try:
        for epoch in range(epochs + 1):
            train_loss = 0.0
            if epoch:
                policy.set_training_mode(True)
                order = train_ids[torch.randperm(len(train_ids), device=model.device)]
                for batch in order.split(batch_size):
                    prediction = policy.get_distribution(obs[batch]).mode()
                    loss = torch.nn.functional.mse_loss(prediction, actions[batch])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
                    optimizer.step()
                    train_loss += loss.item() * len(batch)
                train_loss /= len(train_ids)
            if epoch % eval_every and epoch != epochs:
                continue
            policy.set_training_mode(False)
            with torch.no_grad():
                prediction = policy.get_distribution(obs[val_ids]).mode()
                val_loss = torch.nn.functional.mse_loss(prediction, actions[val_ids]).item()
            success = evaluate(model, eval_env, eval_episodes, eval_seed)
            history.append({"epoch": epoch, "train_mse": train_loss if epoch else None,
                            "validation_mse": val_loss, "blocked_success": success})
            print(f"Epoch {epoch}: validation MSE={val_loss:.5f}, "
                  f"blocked success={success:.1%}", flush=True)
            score = (success, -val_loss)
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone()
                              for k, v in policy.state_dict().items()}
        policy.load_state_dict(best_state)
        # Discard old PPO momentum after changing the actor through supervision.
        policy.optimizer.state.clear()
        policy.set_training_mode(False)
    finally:
        eval_env.close()
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=10)
    args = parser.parse_args()
    if not args.demonstrations.is_file() or not args.resume.is_file():
        parser.error("Demonstrations and checkpoint files must exist.")
    if args.output.exists():
        parser.error("Use a new output directory.")
    if min(args.epochs, args.batch_size, args.eval_episodes, args.eval_every) < 1:
        parser.error("Training and evaluation counts must be positive.")
    if args.seed < 0 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Use a nonnegative seed and a finite positive learning rate.")
    with np.load(args.demonstrations, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    validate_data(data)
    torch.set_num_threads(1)
    env = StaticObstacleEnv(pool_size=0)
    try:
        model = PPO.load(args.resume, env=env, device="cpu")
        if (np.any(data["observations"] < env.observation_space.low)
                or np.any(data["observations"] > env.observation_space.high)):
            parser.error("Demonstration observations are outside this environment's bounds.")
        history = clone_actor(
            model, data, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=args.seed,
            eval_episodes=args.eval_episodes, eval_every=args.eval_every,
        )
        args.output.mkdir(parents=True, exist_ok=False)
        model.save(args.output / "best_model")
        with (args.output / "metrics.csv").open("x", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        print(f"Saved PPO-compatible checkpoint to {args.output / 'best_model.zip'}.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
