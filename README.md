# SO-ARM101: learning to reach around an obstacle

This project trains a simulated robot arm to reach a target beside an upright box, then uses the learned policy to plan a reach for a physical Hiwonder SO-ARM101. An operator measures the box center. Starting from the arm's encoder readings, the planner tries several targets in MuJoCo and previews the selected motion before the operator starts replay.

https://github.com/user-attachments/assets/c82a5a10-724d-4ec5-94ad-52fc09d32077

## Results

| Task | Recorded outcome |
| --- | --- |
| Reaching in simulation, obstacle-free | Reached **98 of 100** held-out targets 4–14 cm from the starting gripper tip. Two episodes timed out and none ended in collision. |
| Reaching around a static box in simulation | The stage 1 policy reached **78%** success on the evaluation layouts used during training to select checkpoints. |
| Supervised hardware demo | One physical demo completed. Other attempts included a physical collision and rejected plans after homing. Hardware repeatability and real-world success rate remain unmeasured. |

The [experiment log](EXPERIMENTS.md) records the reach distances, seeds, earlier obstacle policies, and a joint trace from a failed hardware replay.

## How a reach works

1. **Learn a motion policy.** A Gymnasium environment resets the arm to its training home and samples a reachable target. The static obstacle task adds a box near the direct path. Proximal Policy Optimization (PPO), implemented with Stable-Baselines3, trains a neural network to choose five small joint movements per action. The gripper opening stays fixed. Inputs include joint angles, gripper-tip and target positions, and, for the obstacle task, box geometry. The policy earns reward for moving closer. Contact or an unsafe joint state ends the episode.
2. **Choose a target and path.** Given the box position and starting joints, the planner samples candidate targets and simulates the trained policy for each. It rejects trajectories that time out, collide, violate limits, or miss the clearance margin. It prefers a target beyond the box and can fall back to the near side. Within the preferred group of successful paths, it chooses the one with the largest minimum clearance, using duration to break ties.
3. **Preview and replay.** The hardware script displays the selected joint path in MuJoCo. On `go`, it interpolates the saved joint commands, reads the encoders, and checks each move against the model geometry. The hardware planning model places joints at their commanded positions. A physical servo can follow a different path under load. The green target is a visual marker. Simulation success means the modeled tip is within 2 cm of its center.

## Try the simulator

From the repository root, create a Python environment and run the joint sweep. These PowerShell commands use only the checked-in model and meshes.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy mujoco
python -m scripts.simulation.check_simulation
```

The viewer moves each of the five arm joints in turn. The terminal prints joint limits, tracking errors, and the final gripper-tip position in meters. Add `--headless` to run the same sweep without a window.

## Train and evaluate a policy

Install the learning dependencies in the same environment. The commands below train an obstacle-free reaching policy, continue from its best checkpoint on the stage 1 box task, then evaluate that policy on 50 seeded layouts. Each training command needs a new output directory. Git excludes checkpoints and training logs under `outputs/`, so a fresh clone needs training or a compatible checkpoint for policy playback.

```powershell
python -m pip install gymnasium stable-baselines3 tqdm rich
python -m scripts.policy.train_ppo --task reaching --device cpu --timesteps 300000 --output outputs/example-reaching
python -m scripts.policy.train_ppo --task static-obstacle --curriculum-stage 1 --resume outputs/example-reaching/best/best_model.zip --device cpu --timesteps 200000 --output outputs/example-obstacle
python -m scripts.policy.test_policy outputs/example-obstacle/best/best_model.zip --task static-obstacle --curriculum-stage 1 --episodes 50 --seed 3000
```

These commands start a new experiment. The historical runs and their settings are in [EXPERIMENTS.md](EXPERIMENTS.md). Stage 1 samples targets 10–14 cm from the starting tip with at least 4 cm of initial box clearance. To watch an episode, add `--render` and use `--episodes 1` in the evaluation command. Evaluation also prints the box centers of successful episodes.

To try the planner with that checkpoint, supply the box center in world XY meters and the minimum modeled clearance in meters. The planner reports the selected target and clearance or reports that it could not find a path.

```powershell
python -m scripts.planning.plan_reach --checkpoint outputs/example-obstacle/best/best_model.zip --obstacle-xy 0.315 -0.100 --min-clearance 0.002
```

## Physical demonstration

The demo uses HX-30HM servos and the motor drivers from [Hiwonder's LeRobot fork](https://github.com/Hiwonder-official/hiwonder-SoArm-101). Install that checkout in the hardware environment with `python -m pip install -e "<path-to-hiwonder-SoArm-101>[hiwonder]"`. The hardware script expects a compatible stage 1 checkpoint at `outputs/static-curriculum-stage1-continued/best/best_model.zip`, the calibration JSON for the connected follower, a matching joint-zero mapping, and motion limits for that arm. The two JSON files in `scripts/hardware/` describe the demo unit. The mapping is tied to its calibration file by a hash.

Before running the hardware demo, calibrate the connected follower arm using Hiwonder's LeRobot calibration workflow. Pass the resulting JSON to `--calibration`. The demo checks its offsets and limits against the servos and uses `--mapping` to convert encoder readings into MuJoCo joint angles.

```powershell
python -m scripts.hardware.arm_reach --port "<serial-port>" --calibration "<calibration.json>" --mapping "<follower-zero.json>" --motion-config "<follower-home.json>"
```

Replace the placeholders with your port and file paths. Clear the workspace before typing `home`. After homing, place the upright **36 × 36 × 240 mm** box and enter the **center** as X Y Z in world meters. Keep it fixed during preview and replay. The software accepts a final tip error of up to 4 cm, estimated from the encoders and robot model. Physical clearance and endpoint accuracy have not been independently measured. See the [hardware observations](EXPERIMENTS.md#hardware-reach-demo) before replaying a path.

## Where to read the code

| Question | Entry point |
| --- | --- |
| How are rewards, targets, and obstacle layouts defined? | [Reaching environment](scripts/simulation/reaching_env.py) and [static obstacle environment](scripts/simulation/static_obstacle_env.py) |
| How are checkpoints selected and evaluated? | [PPO training](scripts/policy/train_ppo.py) and [seeded evaluation](scripts/policy/test_policy.py) |
| How does the planner rank paths or simulate a return? | [Reach planner](scripts/planning/plan_reach.py) |
| How does a simulated path become servo commands? | [Hardware replay](scripts/hardware/arm_reach.py), [collision checks](scripts/so101/geometry.py), and [Hiwonder control](scripts/so101/hiwonder_control.py) |
| Where do the robot geometry and starting pose come from? | [Model notes](models/SO101/README.md), [MuJoCo scene](models/SO101/scene.xml), and [training home](scripts/so101/training_home.json) |
