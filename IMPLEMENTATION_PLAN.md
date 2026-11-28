# Simulation planner implementation plan

The first demonstration uses measured mock joint positions, one manually placed
obstacle, and the existing Stage 1 static-obstacle policy. The planner must find
a successful outbound trajectory and independently validate a path home.

## Repository audit

- [x] Identify the checkpoint. The archive at
  `outputs/static-curriculum-stage1-continued/best/best_model.zip` records
  635,000 training steps, a 19-value observation, and five actions.
- [x] Identify the observation layout. `StaticObstacleEnv` supplies five
  normalized arm angles, block center XYZ, full block dimensions, two zeros,
  target XYZ, and gripper-tip XYZ. Cartesian values use MuJoCo world coordinates
  in meters. The jaw-obstacle variant assigns different meanings to three slots.
- [x] Verify the simulator home state. In training order, shoulder pan, shoulder
  lift, elbow flex, wrist flex, wrist roll, and gripper start at
  `[0, -0.13, 0.025, 0.93, 0, 0]` radians. The environment sets velocities
  to zero and initializes the actuator targets to the joint positions.
- [x] Inspect action conversion. Every 50 ms, `ReachingEnv.step` converts the
  five policy outputs into bounded increments from measured joint angles and
  ramps the actuator commands over physics steps. Reuse this method for rollouts.
- [x] Inspect the collision model. The scene includes wrist-camera and mount
  collision meshes. The overhead rendering camera is a separate scene object.
- [x] Locate the Hiwonder driver. The local simulation environment resolves
  `lerobot` to Hiwonder's fork, which includes its motor bus and sync read/write
  SDK. The environment has MuJoCo, Gymnasium, and Stable-Baselines3. ROS 2
  Python bindings were unavailable in that interpreter.
- [x] Verify deterministic inference with the saved policy. Two rollouts from
  the explicit seed-3000 scene produced identical actions and joint positions
  over 110 control steps and reached the target within the 2 cm tolerance.

## Standalone planner

- [x] Add `StaticObstacleEnv.reset_for_planning` with explicit measured joints,
  target XYZ, and obstacle XY. This first version uses the training block's
  fixed height, dimensions, and orientation. The caller supplies home tolerance
  in radians, with exact home as the default for simulation. An accepted scene
  passes initial contact, clearance, observation bounds, and target distance
  checks. Reachability and clearance across the motion still require validation.
- [x] Verify reset equivalence in the simulation environment.
  At seed 3000 the initial observation, reset diagnostics,
  first step result, and resulting joint positions matched the training path.
  Separate checks covered measured-home offsets and rejection of an obstructed
  start followed by a successful training reset.
- [x] Generate reachable Stage 1 targets near the obstacle using seeded
  joint-pose samples. Prefer targets beyond the obstacle's far edge along the
  horizontal direction from the starting gripper tip to the obstacle center.
  Check that the final simulated tip also clears that edge. Try other targets
  if every far-side reach fails.
- [x] Load the checkpoint once and run deterministic candidate rollouts. Record
  timed joint positions, execution commands, and minimum obstacle clearance
  throughout each rollout, including physics substeps.
- [x] Reject collision, limit, clearance, timeout, and target failures. Rank
  accepted trajectories by minimum clearance. Report failure when none qualify.
- [x] Validate the return in MuJoCo from supplied end joint positions and
  velocities. Check the current block XY against the outbound snapshot, retime
  the reversed measured joint positions, and validate the approach to the
  original training home. Check contact, clearance, joint state, and speed at
  each physics step, including the final hold.

### Running the outbound planner

Supply the block center in world XY and the required simulation clearance in
meters. For example, this scene uses a 2 mm demonstration margin.

```powershell
python -m scripts.planning.plan_reach --obstacle-xy 0.29336519 -0.09268185 --min-clearance 0.002
```

With the 635k checkpoint this selected a far-side target at approximately
`[0.2528, -0.1110, 0.1500]` m, with 10.3 mm minimum clearance over 5.5 s.
A block at `[0.29, 0.02]` m failed the start check. With the first layout and
a one-step rollout limit, all four candidates timed out. Both failure cases
returned an empty plan. Physical clearance requirements still need measurement.

### Validating the return

```powershell
python -m scripts.planning.plan_reach --obstacle-xy 0.29336519 -0.09268185 --min-clearance 0.002 --validate-return
```

The command uses the selected reach's simulated endpoint feedback and assumes
the manually placed block stays in place. The validator accepts independent
joint measurements and a current obstacle observation. For the fixed training
block it requires an exact XY match. Camera measurement tolerances remain a
hardware integration task.

Return commands use the outbound command-speed cap and stay inside joint
limits. Simulated positions near a soft stop can exceed a command bound within
the environment's 0.001 rad state tolerance. The validator limits the command
and simulates the resulting motion. At home it requires all six joints within
0.01 rad and speeds at or below 0.01 rad/s for 0.1 s, with a one-second
settling timeout. These are simulation settings.

The example returned in 5.67 s with 10.29 mm minimum clearance and maximum
home error of 0.000513 rad. Shifting the reported obstacle by 1 cm rejected the
return before the first simulation step. A separate run offset the measured
shoulder pan by 0.003 rad, verified the first trajectory point matched that
measurement, and reached settled home with increasing trajectory timestamps.

## ROS 2 and hardware milestones

- [x] Verify ROS 2 Jazzy and Python 3.12 together. With the Jazzy underlay
  sourced, the project environment imported rclpy, MuJoCo, Gymnasium and
  Stable-Baselines3. The simulator and mock robot both read the six ordered
  joint names and positions from `scripts/so101/training_home.json`. The
  standalone planner uses the same definition through its environment.
- [x] Implement mock joint feedback, robot state and a cancellable homing
  action. In the clear mock workspace the arm starts with shoulder pan 0.1 rad
  above home. The action requires clear-workspace confirmation for motion
  and traverses that single joint segment at up to 0.05 rad/s. READY requires
  all six joints within tolerance and stationary for 0.1 s.
- [x] Build the generated ROS interfaces and run `test_mock_homing.py`
  against the mock action server. The operator confirmed the workspace build
  and homing test passed.
- [x] Add the planner's `/planner/check_start` service. It checks the age
  of both feedback topics, READY and fault status, and the position and
  velocity of all six named joints. The verifier supplies measured joints in
  training order. The ROS integration test rejected the start before homing,
  accepted it after settling, and rejected it again after feedback stopped.
- [x] Connect the stamped mock block center and the standalone PPO planner
  to `/start_demo`. Snapshot measured home and obstacle coordinates, run the
  Stage 1 candidate search in a background worker, then publish the selected
  timed actuator targets on `/planned_trajectory` for inspection.
- [x] Add separately authorized mock trajectory execution. The planner
  rechecks robot and obstacle feedback, consumes one selected plan per
  authorization and sends a cancellable `FollowJointTrajectory` goal to the
  mock robot. The robot publishes timed joint feedback and finishes in
  `AT_TARGET`. The single-terminal integration test checks the final six
  positions, rejection after an obstacle change, and cancellation during motion.
- [x] Connect `/return_home` to independent MuJoCo validation. The request
  uses the executed outbound rollout, the current measured joint positions
  and velocities, and the unchanged block. Only a validated trajectory goes
  to the mock return action. Home completion requires all six joints to settle.
- [ ] Implement camera, planner, and robot nodes with mock obstacle input,
  measured mock joint feedback, cancellable homing and trajectory actions,
  and explicit operator authorization for execution.
- [ ] Exercise the full mock cycle, including stale inputs, planning failures,
  cancellation, tracking faults, and independent return validation.
- [ ] Before hardware integration, verify the HX-30HM controller interface,
  motor IDs, baud rate, joint directions, offsets, limits, and measured tracking
  performance. Measure camera intrinsics, camera-to-mount transform, table
  plane, and stick geometry for mapping into the robot base frame. Hardware
  calibration and physical execution are pending.

### Physical follower feedback

- [x] Add an opt-in encoder reader to `robot_node`. Raw mode reads and checks
  the follower calibration registers at all six IDs. Mapped mode uses two
  measured joint references per axis and publishes positions in radians.
  Register reads and torque-preserving disconnect were exercised with the
  Hiwonder fork and simulated serial replies.
- [x] Read all six follower IDs and compare the programmed registers with the
  follower calibration. Jog the physical wrist and verify measured displacement
  and torque release.
- [x] Implement opt-in six-axis ROS control and exercise homing, outbound
  motion, return, cancellation, operator stop and fault handling on simulated
  HX-30HM servos using the Hiwonder SDK.
- [ ] Verify directions and home references for all six physical joints,
  measure working limits and validate the clear-workspace home approach.
  Measure multi-joint tracking and stopping on the actual follower.

### Shared home and mock robot verification

The Stage 1 seed-3000 reset produced byte-identical observations, positions,
velocities and commands before and after extracting the shared pose. The
635k checkpoint reproduced the 5.50 s outbound and 5.67 s return examples.
In the obstacle-free reaching scene, all 1,001 evenly spaced poses on the
predefined mock homing segment passed MuJoCo contact and joint-limit checks.
This check covers the ideal mock path only.

Jazzy parsed both custom interface definitions. The Python package passed
the ROS flake8 and pep257 checks. An in-process check with ROS transport
doubles exercised feedback, clear-workspace rejection, concurrent goal
rejection, cancellation, retry, settling and the already-home case.

### Mock outbound reach verification

The ROS service and topic integration test uses the local 635k checkpoint.
It published a trajectory from shuffled joint feedback with a 0.001 rad
wrist-roll offset and preserved that measurement in the first trajectory
point. Timestamps increased from zero. The blocked layout at
`[0.29, 0.02]` m produced a planning failure. A stale feedback stream
prevented admission, and moving the obstacle during a later search
invalidated that result. Only the successful request published a trajectory.
The test exercises the ROS interfaces in one process with mock measurements.

### Mock return verification

The one-terminal `round_trip` case homes, plans, executes the reach and
explicitly requests a return. A 0.003 rad shoulder-pan offset in the live
endpoint feedback appears in the first return command. The mock reaches all
six training-home positions, reports zero velocity and transitions to READY.
In the `moved_at_target` case, moving the observed block by 1 cm and then
restoring it rejects the return and leaves the arm stationary at the endpoint.
