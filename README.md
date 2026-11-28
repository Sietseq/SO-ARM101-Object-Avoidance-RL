# RL-SO-ARM101

Simulation research on reaching and static-obstacle avoidance for the
Hiwonder SO-ARM101 arm. The project uses MuJoCo, Gymnasium, and PPO with
Stable-Baselines3. See [PROJECT_SPECS.md](PROJECT_SPECS.md) for the original
sim-to-real goals and [EXPERIMENTS.md](EXPERIMENTS.md) for the recorded
training history and results.

## Current project direction

The work progressed from fixed-start reaching on closer targets to randomized
reaches, then static obstacle avoidance. Targets are currently limited to
4–14 cm from the initial end-effector position: the 4–20 cm range proved too difficult on the
longer reaches, while the 4–14 cm evaluation solved 98/100 targets with no
collisions. The latest recorded static curriculum run is
`static-curriculum-stage1-continued`; its recorded evaluation peaked at 78%.
The experiment log distinguishes those training evaluations from held-out
standalone evaluations.

## Repository layout

- `scripts/hardware/` runs the physical reach demo and checks follower motion.
- `scripts/simulation/` contains the MuJoCo environments and model check.
- `scripts/planning/` finds obstacle-avoiding reaches and validates return paths.
- `scripts/policy/` trains PPO policies and evaluates saved checkpoints.
- `scripts/so101/` contains the shared Hiwonder helpers, collision checks, and training home pose.
- `models/SO101/`: SO-ARM101 model files and assets.
- `PROJECT_SPECS.md`: project specification.
- `EXPERIMENTS.md`: results and experiment decisions.
- `ROS2_COMMANDS.md`: ROS 2 environment, build, run, and inspection commands;
  it also marks planned interfaces that do not exist yet.

Training outputs are written under `outputs/` and are ignored by Git. Keep
machine-specific paths, editor settings, credentials, and local environment
configuration out of committed files.

## Hardware and LeRobot dependency

This project uses [Hiwonder's LeRobot fork](https://github.com/Hiwonder-official/hiwonder-SoArm-101).
The fork provides the Hiwonder HX-30HM motor drivers required by this arm.
Preserve this dependency when configuring robot communication. The Python
package is still named `lerobot`.

## Local development

Use Python 3.12 or later in a dedicated environment. Install the local
Hiwonder LeRobot checkout in editable mode with its `hiwonder` extra:

```powershell
python -m pip install -e "<path-to-hiwonder-SoArm-101>[hiwonder]"
```

The project also uses MuJoCo, Gymnasium, Stable-Baselines3, and OpenCV. A
complete dependency setup is not yet defined in this repository. Training
uses the simulator; it does not require connecting to the physical arm.

## Reaching

For a physical reach, set `FOLLOWER_CALIBRATION` to the follower's calibration
JSON and run `python -m scripts.hardware.arm_reach` from the repository root in the project environment.
It uses the local joint mapping and motion limits in `outputs/` (see
`--help` for overrides). Clear the workspace and type `home`. After homing,
place the upright 36 × 36 × 240 mm box and enter its measured center as three
space-separated world coordinates in meters. The script plans from encoder
feedback and asks for `go` when it finds a reach beyond the box. Keep the
box fixed during replay. Ctrl+C ends the demo and requests a measured hold.
The home tolerance defaults to 3° (`--home-tolerance-deg`).

The default task is reaching without obstacles. Training and evaluation
sample targets 4–14 cm from the initial end-effector position. Reachability
and joint-limit checks still apply; the success tolerance is 2 cm.

```powershell
python -m scripts.policy.train_ppo --task reaching --device cpu --timesteps 300000 --output outputs/ppo-reaching-seed0
python -m scripts.policy.test_policy outputs/ppo-reaching-seed0/best/best_model.zip --task reaching --episodes 100 --seed 1000
```

To continue training a reaching checkpoint:

```powershell
python -m scripts.policy.train_ppo --resume outputs/ppo-reaching-seed0/best/best_model.zip --device cpu --learning-rate 0.00003 --timesteps 50000 --eval-freq 5000 --output outputs/ppo-reaching-continued
```

### Hardware reach status

The physical reach demo previews the planned joint path in MuJoCo before
replay. Its hardware target tolerance is 4 cm to account for the arm's limited
repeatability; this is looser than the simulator's 2 cm success tolerance.
The follower home configuration uses a maximum speed of 0.50 rad/s. Treat
these settings as experimental: homing and replay have not been validated
across repeated runs, and hardware joint commands can diverge from the
simulated policy trajectory. Inspect the preview, keep the workspace clear,
and supervise the arm during every replay. See
[EXPERIMENTS.md](EXPERIMENTS.md#hardware-reach-demo) for observed issues.

## Static obstacle avoidance

Train the static-block task with curriculum stage 1, or use
`--task jaw-obstacle` to provide the vector from the moving jaw's closest
surface point to the block's closest surface point. The vector uses world
coordinates in meters and occupies the three block-dimension observation
slots. At contact or overlap it is zero. Stage `1` is the curriculum used so
far; stage `0` retains the original sampler for comparison.

```powershell
python -m scripts.policy.train_ppo --task static-obstacle --curriculum-stage 1 --device cpu --timesteps 200000 --output outputs/static-curriculum-stage1
python -m scripts.policy.test_policy outputs/static-curriculum-stage1/best/best_model.zip --task static-obstacle --curriculum-stage 1 --episodes 50 --seed 3000
```

To continue a static-obstacle policy, add `--resume` with its checkpoint
path. Keep the same task and curriculum stage for training and evaluation.
The training callback selects checkpoints by overall success rate, with
mean reward breaking ties.

## Hardware support

Hardware deployment is not the current training workflow. Before connecting
to a physical arm, verify the Hiwonder fork, calibration, joint ordering,
limits, camera mapping, and independent safety procedures described in
[PROJECT_SPECS.md](PROJECT_SPECS.md).
