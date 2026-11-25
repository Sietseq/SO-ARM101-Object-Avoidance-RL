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
