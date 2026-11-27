# SO-ARM101 experiment log

All results below are from simulation. Training evaluations use a fixed set
of layouts to select checkpoints. Standalone evaluations use the listed
seeds. Changes to the sampler can change the layout associated with a seed.

## Reaching progression

| Stage | Run and setup | Result and decision |
| --- | --- | --- |
| Fixed-start baseline | `ppo-fixed-start-seed0`; held the robot start pose constant and varied the target. | Training evaluation reached 90% at 300k total steps. The saved checkpoint's training evaluation had reached 100% at 140k; the later 90% is the final recorded evaluation, not the peak. |
| Expand the reach range | `ppo-curriculum-4-20cm-cpu-seed0`; randomized start configurations and targets 4–20 cm from the initial end-effector position. | The 360k checkpoint solved 25/30 held-out episodes on seeds 1000–1029. All five failures timed out. Long reaches sometimes got close to the target and then moved away; clipping and joint limits were considered, but neither was confirmed as the cause. |
| Check performance on far targets | `ppo-far-mixed-seed0` and `ppo-far-mixed-no-saturation-seed0`; both evaluated on seeds 2000–2099. | At 455k steps, they solved 90/100 and 91/100. Both solved all 68 shorter reaches; on the 32 longest reaches they solved 22 and 23. The action-saturation penalty did not show a useful advantage in this comparison and was removed. |
| Cap target distance at 14 cm | Evaluated the 360k checkpoint on targets 4–14 cm away, seeds 2023–2122. | Solved 98/100 with zero collisions. Both timeouts were robot-right targets near the outer edge. The 20 cm maximum was too demanding for the intended task, so the project adopted a 14 cm cap. |

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

| Run | Training and result |
| --- | --- |
| `ppo-static-obstacle-finetune-seed0-v3` | Fine-tuned from a reaching checkpoint. After 500k additional steps, fixed evaluation improved from 4/20 to 12/20. In the final 1,000 training episodes, 565 ended in collision, including 221 on the first action. This motivated the start-clearance change. |
| `ppo-static-obstacle-clearance1cm-seed0` | Continued from v3 with 1 cm start clearance. The compared checkpoint was the best available after 5k additional steps. |
| `ppo-jaw-obstacle-seed0` | Replaced the fixed block-dimension observation with the vector from the moving jaw's nearest surface to the block's nearest surface. After about 100k additional steps from v3, fixed evaluation changed from 11/20 to 12/20. |
| `ppo-jaw-curriculum-seed0` | Tried target offsets of 4, 2, then 0 cm between the block center and direct path, advancing after 30 successes in a batch of 50 rightward reaches. A reported 4 cm batch solved 17/50. Hard evaluation peaked at 12/20 through 60k additional steps, matching the starting score. |
| `static-curriculum-stage1-continued` | Continued static-obstacle training with curriculum stage 1. The run's recorded cumulative training evaluations began at 460k steps (61%), reached 78% at 645k, and ended at 78% at 660k. This is the current working result; it uses a different curriculum and evaluation setup from the earlier matched comparison. |

### Matched evaluation on 1 cm clearance layouts

Seeds 3000–3099. Each checkpoint was evaluated with its own observation
format. All four solved 3/5 center targets.

| Checkpoint | Overall | Left | Right | Collisions | Timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 best | 39/100 | 26/53 | 10/42 | 58 | 3 |
| Clearance fine-tune, interim best | 41/100 | 28/53 | 10/42 | 56 | 3 |
| Jaw best | 43/100 | 30/53 | 10/42 | 55 | 2 |
| Jaw final | 43/100 | 32/53 | 8/42 | 56 | 1 |

Across the 42 rightward reaches, the jaw hit first in 23 episodes for v3, 22
for jaw best, and 23 for jaw final. All three hit the block with the jaw on
the same 22 layouts. Median collision time was step four for each checkpoint.
Seed 3006 reproduces a jaw collision on step four with all three.

These matched results describe the earlier checkpoints and should not be
read as results for `static-curriculum-stage1-continued`. The latest run's
saved progress records its curriculum evaluation; a separate held-out
evaluation is not recorded here.
