# RL-SO-ARM101

Sim-to-real reinforcement learning for vision-based obstacle avoidance
with a Hiwonder SO-ARM101 arm. See [PROJECT_SPECS.md](PROJECT_SPECS.md).

## Hardware and LeRobot dependency

This project uses [Hiwonder's LeRobot fork](https://github.com/Hiwonder-official/hiwonder-SoArm-101).

The fork provides the Hiwonder HX-30HM motor drivers required by this
arm. Preserve this dependency when configuring robot communication.
The Python package is still named `lerobot`.

## Local development

Use Python 3.12 or later in a dedicated environment. Install the local
Hiwonder LeRobot checkout in editable mode with its `hiwonder` extra:

```powershell
python -m pip install -e "<path-to-hiwonder-SoArm-101>[hiwonder]"
```

The project specification also calls for MuJoCo, Gymnasium,
Stable-Baselines3, and OpenCV. A complete dependency setup is not yet
defined in this repository.

Keep machine-specific paths, editor settings, credentials, and local
environment configuration out of committed files.

## Stage A PPO training

Use an environment containing MuJoCo, Gymnasium, PyTorch, and
Stable-Baselines3. Training uses only the simulator. The default task is
Stage A reaching without obstacles.

```powershell
python scripts/train_ppo.py --output outputs/ppo-stage-a-seed0
```

Device selection defaults to `--device auto`, which uses CUDA when
available and otherwise uses CPU. Select a device explicitly with:

```powershell
python scripts/train_ppo.py --device cuda --output outputs/ppo-stage-a-gpu
python scripts/train_ppo.py --device cpu --output outputs/ppo-stage-a-cpu
```

Check CUDA availability in the active Python environment:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

For a short integration check that includes a PPO update and evaluation:

```powershell
python scripts/train_ppo.py --timesteps 128 --n-steps 128 --batch-size 64 --eval-freq 128 --eval-episodes 2 --output outputs/ppo-smoke
```

Each run requires a new output directory. Outputs include training
episode diagnostics, PPO metrics, evaluation results, the best model
by evaluation success rate (mean reward breaks ties), and the final model.
PPO rounds training up to complete rollouts. A smoke test does not establish
policy quality.


## Stage B static obstacle

The separate `StaticObstacleEnv` extends the reaching environment. The
starting pose and box position stay fixed while target positions vary,
within the existing 4-20 cm distance range. The visible box is 2 x 2 x 4 cm;
its footprint is an infinite-height exclusion zone across the arm's reachable
height range. Contacts against any modeled robot collision geometry end
the episode. Sampled joint-space detours validate layouts during pool
generation; they are not supplied to the policy.

Training precomputes 512 layouts on first reset and samples from that pool
for subsequent episodes. Static-obstacle training defaults to about 80%
clear-route targets and 20% blocked targets. Clear-route targets have
a validated direct joint path; blocked targets have a sampled detour.
Increase `--blocked-fraction` from `0.2` to `0.5`, then `1.0`, in separate
resumed runs as avoidance improves. Callback evaluation uses only blocked
targets in a separate 128-layout pool with a different generation seed.
Pool construction prints progress and adds a startup delay; pools stay in
memory and rebuild for each run.
Standalone evaluation with `test_policy.py` generates fresh targets for
each seeded episode to check generalization beyond the training pool.

The 19-value observation and five-joint action interfaces are unchanged.
The obstacle fields now contain its normalized overhead-camera bounding
box and zero temporal changes. Projection assumes a square camera image.

Continue from a reaching checkpoint in a new output directory:

```powershell
python scripts/train_ppo.py --task static-obstacle --blocked-fraction 0.2 --resume outputs/ppo-curriculum-4-20cm-cpu-seed0/best/best_model.zip --device cpu --output outputs/ppo-static-mixed-seed0
```

Evaluate on the same task (including when testing a reaching baseline):

```powershell
python scripts/test_policy.py outputs/ppo-static-mixed-seed0/best/best_model.zip --task static-obstacle --episodes 100 --seed 10000 --render
```


## PPO fine-tuning after behavior cloning

Static-obstacle reaching uses a success bonus of 50; Stage A keeps its
bonus of 10. Collision penalties and time costs retain their existing values.
Checkpoint selection prioritizes success rate, with mean reward breaking
ties. The initial policy is evaluated and saved before learning begins.
Every evaluation uses the same seeded scenarios for a consistent comparison.

For a shorter fine-tuning experiment from the preserved BC checkpoint:

```powershell
python scripts/train_ppo.py --task static-obstacle --blocked-fraction 0.2 --resume outputs/bc-static-seed0/best_model.zip --device cpu --learning-rate 0.00003 --action-std 0.1 --target-kl 0.01 --timesteps 50000 --eval-freq 5000 --eval-episodes 100 --output outputs/ppo-after-bc-gentle-seed0
```

The optional learning-rate and KL settings override checkpoint values.
The action standard deviation initializes exploration noise; PPO can
subsequently change it. Without these options, resumed runs keep their
saved settings. Compare success rates across runs; the changed reward
scale makes old and new mean rewards different.
