# SO-ARM101 experiment log

The reaching and obstacle-policy tables report simulation runs. Training evaluations reuse fixed layouts to select checkpoints. Standalone evaluations use the seeds listed with each result. A seed identifies a layout only for the sampler version that generated it. The final section records observations from the physical arm.

## Reaching progression

| Stage                            | Run and setup                                                                                                                                                                         | Result and decision                                                                                                                                                                                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fixed-start baseline             | `ppo-fixed-start-seed0`; held the robot start pose constant and varied targets over the initially shorter distances. The exact distance bounds are not recorded in the run artifacts. | Training evaluation reached 90% at 300k total steps. The saved checkpoint's training evaluation had reached 100% at 140k; the later 90% is the final recorded evaluation, not the peak.                                                                    |
| Expand the reach range           | `ppo-curriculum-4-20cm-cpu-seed0`; randomized start configurations and targets 4–20 cm from the initial end-effector position.                                                        | The 360k checkpoint solved 25/30 held-out episodes on seeds 1000–1029. All five failures timed out. Long reaches sometimes got close to the target and then moved away; clipping and joint limits were considered, but neither was confirmed as the cause. |
| Check performance on far targets | `ppo-far-mixed-seed0` and `ppo-far-mixed-no-saturation-seed0`; both evaluated on seeds 2000–2099.                                                                                     | At 455k steps, they solved 90/100 and 91/100. Both solved all 68 shorter reaches; on the 32 longest reaches they solved 22 and 23. The action-saturation penalty did not show a useful advantage in this comparison and was removed.                       |
| Cap target distance at 14 cm     | Evaluated the 360k checkpoint on targets 4–14 cm away, seeds 2023–2122.                                                                                                               | Solved 98/100 with zero collisions. Both timeouts were robot-right targets near the outer edge. The 20 cm maximum was too demanding for the intended task, so the project adopted a 14 cm cap.                                                             |

Robot-left and robot-right are measured relative to the arm's initial
horizontal direction from shoulder to gripper. These labels describe target
position, not camera orientation.

## Static obstacle setup

- An early vision-based version supplied an approximate obstacle image box.
  It showed little useful avoidance, but there was no matched numerical
  comparison. The task then switched to obstacle center XYZ and dimensions in
  the robot's world frame.
- Uniform obstacle placements often left the direct reach clear. The sampler
  was changed to prefer feasible placements near the direct start-to-target
  segment, while checking that both endpoint poses were collision-free.
- Collision-free start poses could still begin almost touching the block. A
  1 cm minimum surface gap was added for the original sampler. Ten reset
  checks on seeds 1000–1009 had a smallest measured gap of 1.094 cm.
- The block is a static box. Its collision geometry spans the arm's operating
  height, while the task observation reports its world-frame position and
  dimensions. The jaw-observation variant uses the moving jaw's nearest
  surface-to-surface vector in the three dimension slots.

## Static obstacle policy experiments

| Run                                      | Training and result                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ppo-static-obstacle-finetune-seed0-v3`  | Fine-tuned from a reaching checkpoint. After 500k additional steps, fixed evaluation improved from 4/20 to 12/20. In the final 1,000 training episodes, 565 ended in collision, including 221 on the first action. This motivated the start-clearance change.                                                               |
| `ppo-static-obstacle-clearance1cm-seed0` | Continued from v3 with 1 cm start clearance. The compared checkpoint was the best available after 5k additional steps.                                                                                                                                                                                                      |
| `ppo-jaw-obstacle-seed0`                 | Replaced the fixed block-dimension observation with the vector from the moving jaw's nearest surface to the block's nearest surface. After about 100k additional steps from v3, fixed evaluation changed from 11/20 to 12/20.                                                                                               |
| `ppo-jaw-curriculum-seed0`               | Tried target offsets of 4, 2, then 0 cm between the block center and direct path, advancing after 30 successes in a batch of 50 rightward reaches. A reported 4 cm batch solved 17/50. Hard evaluation peaked at 12/20 through 60k additional steps, matching the starting score.                                           |
| `static-curriculum-stage1-continued`     | Continued static-obstacle training with curriculum stage 1. The run's recorded cumulative training evaluations began at 460k steps (61%), first reached 78% at 635k, and ended at 78% at 660k. This is the current working result; it uses a different curriculum and evaluation setup from the earlier matched comparison. |

## Hardware reach demo

One supervised reach demo completed on the Hiwonder arm. An operator measured the box center, inspected the simulated path, and authorized replay. A physical collision occurred on another attempt. The number of comparable hardware trials was not recorded, so a hardware success rate cannot be calculated.

### What the replay log showed

An earlier replay stopped with `Replay pose in collision` after waypoint 67 of 115. That message came from the model's pose check. At waypoint 42 the shoulder-lift trace read:

| Planned angle | Command sent | Encoder reading | Reading minus command | Reading minus plan |
| ------------: | -----------: | --------------: | --------------------: | -----------------: |
|        −38.6° |        −6.3° |           −6.3° |                  0.0° |             +32.3° |

The shoulder closely tracked the command at that instant. The large discrepancy was already present between the plan and the command. The printed `err` field measures encoder reading minus command, so a small value there can coexist with a visibly different arm pose.

### What remains uncertain

- **Starting pose.** Homing from an extended pose sometimes left the measured start too close to the modeled obstacle for planning. A box center that succeeded once could be rejected on a later run.
- **Endpoint accuracy.** The simulator accepts a gripper tip within 2 cm of the target. Hardware replay accepts a model-estimated error up to 4 cm using final encoder readings. One attempt reported 2.7 cm. Independent tip measurements and repeatability across placements are still missing.
- **Replay changes.** The current implementation streams interpolated waypoints and the demo configuration caps each joint at 0.50 rad/s. The earlier trace explains why planned, commanded, and measured angles are logged separately. Repeated trials with the revised replay are still needed to quantify tracking and clearance.
