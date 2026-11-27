# SO-ARM101 — ROS 2 Vision-Guided Reinforcement Learning and Obstacle Avoidance

## 1. Project objective

Extend the existing `RL-SO-ARM101` reinforcement learning repository into a modular ROS 2 robotic system capable of autonomously detecting an obstacle, generating a nearby reaching target, planning a collision-free trajectory using a pretrained reinforcement learning policy, and executing that trajectory on a physical robot.

The physical robot is based on the SO-ARM101 mechanical design and uses:

- Hiwonder servo motors.
- A wrist-mounted camera.
- The same mechanical configuration represented in the existing MuJoCo model.
- The same fixed starting joint configuration used during reinforcement learning training.

The final demonstration should perform the following sequence:

1. Move the physical robot into the original training starting configuration.
2. Verify the starting configuration using measured motor positions.
3. Capture an image of the workspace.
4. Detect a wooden stick.
5. Map the obstacle into the robot base coordinate system.
6. Automatically generate multiple nearby reaching targets.
7. Simulate PPO trajectories for the candidate targets in MuJoCo.
8. Reject unsuccessful or unsafe trajectories.
9. Select the safest successful trajectory.
10. Execute the selected trajectory on the physical robot.
11. Report the execution result.
12. Return the robot to its starting configuration using a validated return trajectory.
13. Repeat with a different obstacle placement.

The objective is a reliable student demonstration combining:

- Computer vision.
- Camera calibration.
- Spatial mapping.
- Robot kinematics.
- Reinforcement learning.
- Curriculum learning.
- Simulation-based motion planning.
- ROS 2 communication.
- Sim-to-real transfer.
- Physical robot control.

The PPO policy does not need to achieve 100% success. The planning system should compensate for some of its limitations through candidate target generation and simulation-based trajectory validation.

## 2. Existing repository and experimental results

The existing repository contains:

- A MuJoCo model of the SO-ARM101.
- Wrist-mounted camera collision geometry.
- A Gymnasium reaching environment.
- A static obstacle avoidance environment.
- PPO training and evaluation scripts.
- Pretrained reaching policies.
- Pretrained obstacle avoidance policies.
- A geometric obstacle avoidance curriculum.

Relevant files include:

- `scripts/reaching_env.py`
- `scripts/static_obstacle_env.py`
- `scripts/train_ppo.py`
- `scripts/test_policy.py`
- `scripts/check_curriculum.py`
- `models/SO101/so101_new_calib_camera.xml`

Verify these paths and the current repository structure before implementation.

### Existing trained policy

Current obstacle avoidance checkpoint:

`outputs/static-curriculum-stage1-continued/best/best_model.zip`

The latest reported best checkpoint was saved at 635,000 total training steps.

The policy was initialized from an obstacle-free reaching policy trained on target distances of 4–20 cm.

Obstacle avoidance training uses the stage 1 curriculum:

- Target distance: 10–14 cm.
- Minimum initial arm-to-obstacle clearance: 4 cm.
- Obstacles positioned primarily near the reaching target.
- Randomized target positions.
- Randomized obstacle positions.
- Fixed initial robot joint configuration.

Previous evaluation results:

| Metric                  | 590k checkpoint | 635k checkpoint |
| ----------------------- | --------------: | --------------: |
| Evaluation episodes     |             200 |              50 |
| Success rate            |             85% |             72% |
| Collision rate          |             15% |             28% |
| First-action collisions |              0% |              0% |
| Timeout rate            |              0% |              0% |

These evaluations used different seeds and sample sizes. They should not be interpreted as a direct checkpoint comparison.

The policy has learned an observed three-phase movement strategy:

1. Reposition the arm into an intermediate configuration.
2. Rotate toward the target.
3. Approach the target.

This behavior appears to accommodate the wrist-mounted camera geometry.

Most remaining collisions occur during the initial repositioning movement.

Do not redesign or retrain the PPO policy as part of this implementation unless explicitly requested.

## 3. Hardware configuration

The physical robot uses the SO-ARM101 mechanical design with Hiwonder servo motors rather than the standard Feetech servo configuration.

### 3.1 Hiwonder motors

The hardware implementation must:

- Use the appropriate Hiwonder motor communication protocol or SDK.
- Inspect and reuse existing Hiwonder motor control code where available.
- Not assume compatibility with standard SO-ARM101 Feetech drivers.
- Not assume the standard LeRobot hardware interface is directly compatible.
- Identify the exact Hiwonder servo models.
- Identify the controller board and communication interface.
- Verify motor IDs and baud rate.
- Verify supported motor feedback and command capabilities.

The following hardware parameters must be configurable:

- Serial port or communication device.
- Communication baud rate.
- Motor IDs.
- Motor direction.
- Servo zero-position offsets.
- Servo position conversion factors.
- Joint limits.
- Velocity limits.
- Movement timing.
- Maximum tracking error.
- Communication timeout.

Do not invent these values.

Obtain them from existing repository code, manufacturer documentation, calibration data or physical measurements.

### 3.2 Joint mapping

Establish a consistent mapping between:

- Physical Hiwonder motor IDs.
- ROS joint names.
- MuJoCo joint names.
- PPO action indices.

All ROS joint positions must use radians.

Convert between Hiwonder servo units and radians within the hardware interface layer.

Account for motor mounting orientation, reversed axes and calibration offsets.

### 3.3 Camera hardware

The camera is mounted on the robot arm.

Its position and orientation change as the robot moves.

The camera system requires:

- Camera intrinsics.
- Lens distortion coefficients.
- Camera-to-mount extrinsic calibration.
- Robot forward kinematics.
- A consistent robot base coordinate frame.

Inspect the existing repository for calibration files and camera geometry before introducing new calibration procedures.

Initially, camera-based obstacle mapping will occur only when the robot is stationary at its verified training starting configuration.

This simplifies the perception pipeline while maintaining consistency with the trained policy.

## 4. System architecture

Implement exactly three primary application nodes:

1. `camera_node`
2. `planner_node`
3. `robot_node`

Use ROS 2 with Python and `rclpy`.

Select a ROS 2 distribution compatible with the development operating system and required Python dependencies.

Keep the implementation modular. Supporting Python modules and standard ROS infrastructure are permitted, but do not introduce unnecessary additional application nodes.

### Camera node

Responsible for:

- Image acquisition.
- Wooden stick detection.
- Camera calibration.
- Spatial mapping.
- Obstacle publication.

### Planner node

Responsible for:

- Target generation.
- PPO inference.
- MuJoCo simulation.
- Trajectory validation.
- Trajectory selection.
- Return trajectory validation.
- Demonstration orchestration.

### Robot node

Responsible for:

- Hiwonder motor communication.
- Joint feedback.
- Moving into the original training starting configuration.
- Maintaining robot operational state.
- Trajectory execution.
- Tracking and hardware safety monitoring.
- Controlled stopping.

Nodes must communicate through ROS interfaces instead of directly importing each other's application logic.

The camera node should not know how the robot executes trajectories.

The robot node should not know anything about PPO or reinforcement learning.

The planner node should remain independent of the hardware-specific Hiwonder communication protocol.

## 5. Fixed starting configuration

The original training starting configuration is a mandatory requirement for the initial demonstration.

The PPO policy was trained from a fixed joint configuration. Do not assume that it generalizes to arbitrary initial joint configurations.

### 5.1 Extracting the starting configuration

Inspect the existing reaching and static obstacle environments to identify the exact initial joint positions.

Extract:

- All controlled arm joint positions.
- Gripper position.
- Relevant actuator targets.
- Joint ordering.
- Camera pose associated with the initial configuration.

Do not define a new starting configuration.

Do not assume the starting configuration corresponds to zero radians.

### 5.2 Shared configuration

Maintain a single authoritative definition of the original training starting configuration.

Both the planner node and robot node must use this definition.

Ensure consistency between:

- MuJoCo joint positions.
- ROS joint names.
- Physical Hiwonder servo positions.
- PPO observation ordering.
- Gripper configuration.
- Camera orientation.

Avoid duplicating independent hardcoded home configurations.

If necessary, refactor the existing environment so its initial configuration can be imported or exported without modifying training behavior.

### 5.3 Starting configuration verification

Before beginning a demonstration, verify that:

- The physical robot is connected.
- Joint feedback is available.
- All measured joints are within configurable tolerances of the original training starting configuration.
- The gripper is in the correct configuration.
- The robot is stationary.
- Motor communication is functioning correctly.

The planner must not begin PPO planning from an unverified starting configuration.

If the robot is not at home, request a homing operation.

## 6. Robot homing functionality

The robot node must provide dedicated functionality for moving the physical robot into the training starting configuration.

Suggested internal method:

```python
move_to_home()
```

Suggested ROS action:

`/robot/move_to_home`

Use a ROS action rather than a blocking service so that movement feedback, cancellation and execution results are available.

Define a minimal custom action if no suitable standard action exists.

### 6.1 Homing procedure

The homing operation must:

1. Read current measured joint positions.
2. Determine whether the robot is already at home.
3. Obtain an appropriate validated movement toward home.
4. Execute the movement at a conservative speed.
5. Monitor joint feedback.
6. Monitor tracking error.
7. Monitor hardware communication.
8. Verify that the final measured configuration satisfies home tolerances.
9. Wait until the arm has settled.
10. Report whether homing succeeded.

Do not assume that simultaneously commanding every motor to its home position produces a collision-free movement.

### 6.2 Initial homing strategy

For the initial implementation, perform homing in a confirmed obstacle-free workspace.

This avoids requiring a separate general-purpose motion planner solely for moving into the starting configuration.

The initial demonstration procedure may therefore be:

1. Confirm that the workspace is clear.
2. Move the robot home.
3. Place the wooden stick in the workspace.
4. Begin obstacle detection and planning.

Validate homing trajectories against joint limits, self-collision and relevant workspace geometry.

If the workspace contains an obstacle, only allow homing when a suitable obstacle-aware return trajectory has been validated.

If no safe homing movement can be established, reject the request and require operator intervention.

### 6.3 Robot operational state

The robot node should maintain and publish an operational state.

Suggested states:

- `DISCONNECTED`
- `IDLE`
- `HOMING`
- `READY`
- `EXECUTING`
- `AT_TARGET`
- `RETURNING`
- `STOPPED`
- `ERROR`

The READY state specifically means:

- The robot is connected.
- Valid measured joint positions are available.
- The robot is stationary.
- The robot is within tolerance of the original training starting configuration.
- No execution fault is active.

After completing a reaching trajectory, transition to AT_TARGET rather than READY.

Only transition back to READY after successful homing or return-to-home execution.

### 6.4 Return-to-home functionality

After completing a reaching trajectory, the system should support returning the robot to its original training starting configuration.

For a static obstacle, investigate reversing the previously validated outbound joint trajectory as a candidate return path.

The reversed trajectory must:

- Begin from the actual measured physical robot configuration.
- Be assigned appropriate timestamps.
- Respect joint and velocity limits.
- Be independently validated in MuJoCo.
- Maintain sufficient obstacle clearance.
- Account for the current obstacle configuration.
- Account for physical motor tracking limitations.

Do not simply reverse the PPO actions.

Do not assume a successful forward trajectory guarantees safe reverse execution.

The planner node is responsible for validating the return trajectory.

The robot node is responsible for executing it.

If no safe return trajectory is available, keep the robot stationary and request operator intervention.

Initially, require a separate operator command for returning home. Automatic return can be enabled after validation.

## 7. Camera node

### 7.1 Responsibilities

- Connect to the physical camera.
- Capture images.
- Publish camera images.
- Publish calibration information.
- Detect the wooden stick.
- Estimate obstacle position, orientation and dimensions.
- Transform obstacle measurements into robot base coordinates.
- Publish mapped obstacle geometry.
- Report detection validity and confidence.
- Reject stale or unreliable detections.

### 7.2 Image acquisition

The initial demonstration should capture the obstacle image while the robot is stationary at its original training starting configuration.

The camera node should subscribe to robot state and measured joint positions.

Only accept a planning image when the robot is in the READY state and the corresponding joint measurements are valid.

Allow continuous image publication for visualization and debugging.

However, distinguish between general camera images and obstacle measurements accepted for planning.

### 7.3 Obstacle detection

Initially prioritize a simple, robust detection method.

Possible approaches:

- Color segmentation.
- Background segmentation.
- Contour detection.
- Known geometric features.
- Fiducial markers if necessary.

Avoid introducing a neural network object detector unless simpler methods prove insufficient.

The initial demonstration only requires detecting a single wooden stick in a controlled workspace.

### 7.4 Metric obstacle mapping

If the camera is monocular, do not assume that image coordinates directly provide metric depth.

Investigate using:

- Calibrated camera intrinsics.
- Known table-plane geometry.
- Known obstacle dimensions.
- Ray-plane intersection.
- Perspective geometry.

Estimate the obstacle footprint, orientation and height using an appropriate geometric method.

Account for the fact that visible obstacle surfaces may be elevated above the table plane.

If some dimensions cannot be reliably estimated from an image, allow measured wooden stick dimensions to be configured.

### 7.5 Coordinate transformations

Use a consistent transform chain:

`base_link → arm links → camera_link → camera_optical_frame`

Use robot forward kinematics and TF2 where appropriate.

Associate camera measurements with the measured joint configuration at image acquisition time.

Even though the camera will initially capture planning images from a fixed starting configuration, do not assume that physical motor positioning is perfect.

Use measured joint positions and calibrated camera transformations.

### 7.6 Camera output

Publish obstacle information containing:

- Timestamp.
- Reference coordinate frame.
- Position.
- Orientation.
- Full dimensions.
- Detection confidence.
- Detection validity.

Obstacle coordinates should ultimately be available in the robot base frame.

If detection fails, publish or otherwise communicate an invalid detection rather than silently retaining an old obstacle as current.

## 8. Planner node

The planner is the central application component.

It combines obstacle mapping, measured robot configuration, the pretrained PPO policy and MuJoCo simulation.

### 8.1 Responsibilities

- Subscribe to mapped obstacle information.
- Subscribe to measured joint states.
- Subscribe to robot operational state.
- Verify the original training starting configuration.
- Load the pretrained PPO checkpoint.
- Generate nearby candidate targets.
- Initialize MuJoCo with explicit robot, obstacle and target configurations.
- Run deterministic PPO inference.
- Simulate complete trajectories.
- Check collisions.
- Calculate minimum obstacle clearance.
- Reject unsuccessful trajectories.
- Rank successful trajectories.
- Publish visualization information.
- Request execution of the selected trajectory.
- Generate and validate return-to-home trajectories.

### 8.2 Planning principle

The reinforcement learning policy acts as a candidate motion generator.

MuJoCo acts as a trajectory validation environment.

The physical robot must not directly execute arbitrary PPO actions without prior validation.

The planner should:

1. Verify robot readiness.
2. Obtain a valid mapped obstacle.
3. Generate a candidate target.
4. Simulate the complete PPO trajectory.
5. Determine whether the trajectory succeeds.
6. Determine whether the trajectory satisfies collision and safety constraints.
7. Repeat for additional candidate targets.
8. Select the safest successful trajectory.

If no suitable trajectory exists, report planning failure and do not move the robot.

### 8.3 Demonstration orchestration

The planner node should coordinate the demonstration lifecycle.

Suggested application states:

- `WAITING_FOR_HOME`
- `WAITING_FOR_OBSTACLE`
- `PLANNING`
- `PLAN_READY`
- `EXECUTING`
- `AT_TARGET`
- `PLANNING_RETURN`
- `RETURNING_HOME`
- `COMPLETE`
- `FAILED`

Keep the implementation simple.

Prevent overlapping demonstration requests or multiple simultaneous execution goals.

Long-running planning operations must not prevent stop requests, cancellation or necessary ROS feedback from being processed.

## 9. Robot node

### 9.1 Responsibilities

- Establish communication with Hiwonder motors.
- Read measured motor positions.
- Convert motor positions into joint angles.
- Publish measured joint states.
- Maintain robot operational state.
- Provide homing functionality.
- Receive time-parameterized joint trajectories.
- Execute validated trajectories.
- Monitor trajectory tracking.
- Enforce hardware joint and velocity limits.
- Support cancellation and stopping.
- Handle communication errors.
- Report execution results.

### 9.2 Hardware abstraction

Implement a hardware interface abstraction.

Suggested classes:

- `BaseRobotInterface`
- `HiwonderRobotInterface`
- `MockRobotInterface`

The Hiwonder interface should provide methods such as:

```python
connect()
disconnect()
read_joint_positions()
write_joint_positions()
enable_torque()
disable_torque()
stop()
```

Adapt these methods to the actual Hiwonder hardware capabilities.

Do not assume that torque disabling is always a safe stopping strategy, especially if gravity can cause the arm to fall.

The robot node should interact with hardware through this abstraction.

Only the Hiwonder interface should contain motor-specific communication logic.

Homing is a robot node operation implemented using the hardware interface, not a motor-specific protocol feature.

### 9.3 Mock robot interface

Implement mock hardware for development without a physical robot.

The mock interface should:

- Maintain simulated joint positions.
- Accept trajectory commands.
- Publish joint feedback.
- Simulate homing.
- Simulate successful execution.
- Support cancellation.
- Allow simulated communication failures.
- Allow simulated tracking errors.

This enables ROS integration testing independently of the physical hardware.

### 9.4 Physical execution independence

The robot node must not:

- Load the PPO checkpoint.
- Generate PPO actions.
- Detect obstacles from images.
- Generate reaching targets.
- Choose between candidate PPO trajectories.

These responsibilities belong to the other nodes.

The robot node must independently enforce hardware execution constraints even when a trajectory has already passed planner validation.

## 10. ROS 2 communication interfaces

Use standard ROS message types where possible.

| Interface             | Communication | ROS type                                    |
| --------------------- | ------------- | ------------------------------------------- |
| `/camera/image_raw`   | Topic         | `sensor_msgs/msg/Image`                     |
| `/camera/camera_info` | Topic         | `sensor_msgs/msg/CameraInfo`                |
| `/joint_states`       | Topic         | `sensor_msgs/msg/JointState`                |
| `/robot/state`        | Topic         | Custom robot state message                  |
| `/obstacle`           | Topic         | Custom obstacle message                     |
| `/target_pose`        | Topic         | `geometry_msgs/msg/PoseStamped`             |
| `/planned_trajectory` | Topic         | `trajectory_msgs/msg/JointTrajectory`       |
| `/execute_trajectory` | Action        | `control_msgs/action/FollowJointTrajectory` |
| `/robot/move_to_home` | Action        | Custom homing action                        |
| `/start_demo`         | Service       | `std_srvs/srv/Trigger`                      |
| `/execute_plan`       | Service       | `std_srvs/srv/Trigger`                      |
| `/return_home`        | Service       | `std_srvs/srv/Trigger`                      |
| `/stop_robot`         | Service       | `std_srvs/srv/Trigger`                      |
| `/planner/markers`    | Topic         | `visualization_msgs/msg/MarkerArray`        |
| `/diagnostics`        | Topic         | `diagnostic_msgs/msg/DiagnosticArray`       |

These interfaces are proposed and may be refined during the repository audit.

### 10.1 Custom obstacle message

Create a minimal custom obstacle message.

Suggested definition:

```text
std_msgs/Header header
geometry_msgs/Pose pose
geometry_msgs/Vector3 dimensions
float32 confidence
bool valid
```

Dimensions represent full physical dimensions, not half-extents.

The frame ID and timestamp must be populated correctly.

If additional uncertainty information is required, extend the interface deliberately.

### 10.2 Custom robot state message

Include:

- Timestamp.
- Operational state.
- Hardware connection validity.
- Whether the robot is at home.
- Whether the robot is stationary.
- Whether hardware execution is enabled.
- Current error information.

Use enumerated state constants rather than relying exclusively on arbitrary text strings.

### 10.3 Homing action

Define a custom `MoveToHome.action` if needed.

The action should support:

- A homing request.
- Progress feedback.
- Current joint error relative to home.
- Cancellation.
- Successful or unsuccessful completion.
- Failure reason.

An initial homing action may use a predefined, independently validated trajectory in a confirmed clear workspace.

Obstacle-aware return planning remains the planner node's responsibility.

### 10.4 Trajectory execution action

Use `FollowJointTrajectory` for physical trajectory execution.

The action server should provide:

- Goal acceptance or rejection.
- Execution feedback.
- Measured joint positions.
- Tracking error.
- Cancellation.
- Success or failure results.

A published planned trajectory is for inspection and visualization.

It must not automatically cause physical movement.

Physical movement requires an explicitly accepted action goal and appropriate execution authorization.

### 10.5 Services

Service callbacks should acknowledge or reject requests promptly.

Long-running operations should execute separately, with their progress communicated through state topics, diagnostics or actions.

Do not block stop or cancellation processing while planning or executing.

### 10.6 Units and conventions

Use:

- Meters for positions and dimensions.
- Radians for joint positions.
- Radians per second for joint velocities.
- ROS timestamps.
- Consistent right-handed coordinate frames.

Document all joint names and frame conventions.

## 11. Proposed repository structure

Preserve the existing reinforcement learning repository.

Add a ROS 2 workspace rather than replacing the current training code.

```text
RL-SO-ARM101/
├── models/
│   └── SO101/
├── scripts/
│   ├── reaching_env.py
│   ├── static_obstacle_env.py
│   ├── train_ppo.py
│   ├── test_policy.py
│   └── check_curriculum.py
├── outputs/
├── tests/
├── ros2_ws/
│   └── src/
│       ├── so101_interfaces/
│       │   ├── msg/
│       │   │   ├── Obstacle.msg
│       │   │   └── RobotState.msg
│       │   ├── action/
│       │   │   └── MoveToHome.action
│       │   ├── CMakeLists.txt
│       │   └── package.xml
│       └── so101_demo/
│           ├── package.xml
│           ├── setup.py
│           ├── setup.cfg
│           ├── resource/
│           ├── launch/
│           │   └── demo.launch.py
│           ├── config/
│           │   ├── demo.yaml
│           │   ├── camera.yaml
│           │   ├── robot.yaml
│           │   └── planner.yaml
│           ├── rviz/
│           │   └── demo.rviz
│           ├── test/
│           └── so101_demo/
│               ├── __init__.py
│               ├── camera_node.py
│               ├── planner_node.py
│               ├── robot_node.py
│               ├── common/
│               │   ├── __init__.py
│               │   ├── joint_mapping.py
│               │   └── robot_configuration.py
│               ├── perception/
│               │   ├── __init__.py
│               │   ├── obstacle_detection.py
│               │   ├── calibration.py
│               │   └── spatial_mapping.py
│               ├── planning/
│               │   ├── __init__.py
│               │   ├── simulation.py
│               │   ├── policy.py
│               │   ├── target_generation.py
│               │   ├── trajectory_validation.py
│               │   ├── trajectory_ranking.py
│               │   └── return_planning.py
│               └── hardware/
│                   ├── __init__.py
│                   ├── base_interface.py
│                   ├── hiwonder_interface.py
│                   └── mock_interface.py
├── IMPLEMENTATION_PLAN.md
├── EXPERIMENTS.md
└── README.md
```

Adjust the exact package structure if necessary after inspecting the repository.

Avoid duplicating existing MuJoCo models and RL environment implementations.

Make checkpoint and model paths configurable.

Ensure existing training scripts remain usable.

## 12. Reinforcement learning integration

Preserve compatibility with the trained PPO checkpoint.

### 12.1 Observation space

The existing static obstacle environment is understood to use a 19-element observation vector:

| Observation component          | Elements |
| ------------------------------ | -------: |
| Normalized arm joint positions |        5 |
| Obstacle center coordinates    |        3 |
| Full obstacle dimensions       |        3 |
| Reserved zero-valued fields    |        2 |
| Target coordinates             |        3 |
| End-effector coordinates       |        3 |
| Total                          |       19 |

Verify the exact observation ordering, normalization, units and data types against the repository and checkpoint.

Do not change the observation format.

Reuse existing observation construction code wherever possible.

### 12.2 Action space

The trained policy produces five normalized joint movement actions.

The existing implementation is understood to use:

- A maximum joint increment of 0.02 radians per control step.
- A joint velocity limit of 0.5 radians per second.
- Measured joint positions when constructing incremental commands.

Verify these values against the actual environment implementation.

Reuse existing action handling instead of recreating an independent interpretation of PPO actions.

Preserve the existing gripper configuration.

### 12.3 Checkpoint loading

Load the policy once when initializing the planner.

Make the following configurable:

- Checkpoint path.
- Inference device.
- Deterministic inference.
- Maximum rollout length.

Use CPU inference initially unless another device provides a demonstrated benefit.

Verify that the loaded checkpoint matches the environment's observation and action spaces.

Fail explicitly if compatibility checks fail.

## 13. Explicit MuJoCo environment initialization

The existing environment randomly samples targets and obstacles during reset.

The planner requires explicit initialization.

Extend the existing static obstacle environment with a planning initialization method accepting:

- Initial measured joint positions.
- Target position.
- Obstacle position.
- Obstacle dimensions where supported.
- Obstacle orientation where supported.

Suggested interface:

```python
env.reset_for_planning(
    joint_positions=joint_positions,
    target_position=target_position,
    obstacle_position=obstacle_position,
    obstacle_dimensions=obstacle_dimensions,
)
```

Adapt the interface to the existing codebase.

The method must:

1. Initialize the simulation state.
2. Set the specified robot joint configuration.
3. Set joint velocities consistently with the stationary initial state.
4. Set actuator targets consistently.
5. Set the obstacle geometry.
6. Set the target.
7. Update MuJoCo forward kinematics.
8. Check for invalid initial collisions.
9. Construct the correct policy observation.

The method must not silently randomize or replace the supplied target or obstacle.

Preserve the existing randomized reset behavior used for training and evaluation.

### 13.1 Starting configuration consistency

Before planning, verify that the measured physical configuration is within tolerance of the original training configuration.

Initialize every candidate simulation from the same measured physical configuration.

Do not silently initialize the simulated arm to the exact nominal home configuration if the physical arm is in a different measured configuration.

If the discrepancy exceeds the permitted tolerance, reject the planning request and require homing.

### 13.2 Obstacle representation

The policy was trained with a fixed block-shaped obstacle.

The physical obstacle will be a wooden stick.

Initially, represent the wooden stick using a compatible, conservative collision box.

The collision geometry must enclose the physical obstacle, accounting for:

- Estimated dimensions.
- Orientation.
- Mapping error.
- Calibration uncertainty.
- Required clearance.

If the existing environment only supports axis-aligned obstacles, use an enclosing axis-aligned box initially.

Do not assume that the trained policy generalizes to arbitrary obstacle dimensions or orientations.

The collision geometry may need to be more conservative than the obstacle representation provided to the policy. Document and validate any such distinction.

Do not remove the camera collision geometry.

## 14. Automatic target generation

The obstacle is detected first.

The target is generated automatically based on the mapped obstacle.

The target is not manually specified during the final demonstration.

### 14.1 Candidate generation

Generate approximately 10–20 candidate targets around the mapped obstacle.

Potential candidates can be generated using angular sampling and distance offsets around the obstacle footprint.

Keep the algorithm simple initially.

Candidate targets must satisfy:

- Target distance from the starting end effector between 10 and 14 cm.
- Minimum initial arm-to-obstacle clearance of 4 cm.
- Reachable robot workspace.
- Appropriate target height.
- Sufficient separation from the obstacle.
- No target inside obstacle geometry.
- Compatibility with the stage 1 training distribution where practical.

The initial clearance condition depends on the robot and obstacle configuration, not the target alone.

If it is violated, reject the planning request rather than expecting target generation to resolve it.

### 14.2 Reachability

Validate candidate reachability using available robot kinematics or simulation.

Use existing target-generation and reachability functionality where possible.

Do not assume that every Cartesian position inside a bounding box is reachable.

### 14.3 Candidate diversity

Generate targets on different sides of the obstacle where feasible.

Avoid generating many nearly identical candidates.

Initially use deterministic or seeded candidate generation to improve reproducibility.

Candidate generation should be configurable through YAML parameters.

If no suitable candidate exists, report planning failure.

## 15. Simulation-based trajectory generation

For each candidate target:

1. Reset MuJoCo to the verified measured home configuration.
2. Insert the mapped obstacle.
3. Set the candidate target.
4. Construct the policy observation.
5. Run deterministic PPO inference.
6. Apply the action through the existing environment implementation.
7. Advance MuJoCo.
8. Record joint positions and actuator targets.
9. Record simulation timestamps.
10. Monitor collisions.
11. Monitor minimum obstacle clearance.
12. Repeat until success, collision, unsafe state or timeout.

Record the full trajectory, not just the final joint configuration.

Preserve physics-substep collision checking.

### 15.1 Trajectory data

Each candidate rollout should store:

- Candidate target.
- Initial joint configuration.
- Obstacle configuration.
- Joint position sequence.
- Actuator target sequence.
- Timestamps.
- End-effector trajectory.
- Minimum obstacle clearance.
- Number of control steps.
- Total reward.
- Final target distance.
- Success status.
- Collision status.
- Timeout status.
- Unsafe-state status.
- Collision geometry names when available.

Use structured Python data classes where appropriate.

### 15.2 Collision validation

Check all relevant robot geometry, including:

- Gripper.
- Wrist.
- Forearm.
- Upper arm.
- Camera.
- Camera mount.
- Other relevant links.

Do not rely only on end-effector-to-obstacle distance.

The policy's observed initial repositioning movement can collide even when the final target is reachable.

Reject any trajectory containing a relevant collision.

### 15.3 Minimum clearance

Calculate or estimate the minimum distance between robot geometry and obstacle geometry over the entire trajectory.

Use appropriate MuJoCo geometry-distance functionality.

Check clearance throughout movement, including between commanded trajectory points.

Apply a configurable safety margin derived from measured physical tracking and mapping uncertainty.

Do not invent a safety margin and describe it as experimentally validated.

## 16. Trajectory selection

Only successful trajectories satisfying all validation requirements are eligible.

### 16.1 Rejection criteria

Reject trajectories that:

- Collide with relevant geometry.
- Fail to reach the target.
- Exceed the rollout time limit.
- Violate joint limits.
- Violate velocity limits.
- Contain invalid numerical values.
- Have insufficient obstacle clearance.
- Begin from an invalid robot configuration.
- Depend on stale obstacle measurements.

### 16.2 Ranking criteria

Use hierarchical ranking initially:

1. Successful and valid trajectories only.
2. Prefer the greatest minimum obstacle clearance.
3. Prefer shorter execution duration when clearance is comparable.
4. Prefer smoother joint movements when other criteria are comparable.

Do not introduce another machine learning model or complex optimization algorithm for trajectory ranking.

The ranking logic should be deterministic and independently testable.

### 16.3 Planning failure

If no valid trajectory exists:

- Do not send an execution goal.
- Publish a planning failure result.
- Report rejection reasons.
- Keep the robot stationary.
- Allow the operator to reposition the obstacle and retry.

Do not force the robot to attempt the least unsuccessful trajectory.

## 17. Physical trajectory execution

Convert the selected simulated trajectory into a ROS `JointTrajectory`.

Preserve:

- Joint ordering.
- Joint units.
- Initial joint configuration.
- Relevant control timing.
- Joint limits.

Determine experimentally whether simulated measured joint positions or actuator target positions provide the better hardware execution reference.

Do not assume these are interchangeable.

### 17.1 Execution considerations

Hiwonder motors may have different dynamics from the simulated actuators.

Pay attention to:

- Motor response time.
- Position tracking error.
- Backlash.
- Movement synchronization.
- Servo command timing.
- Velocity limitations.
- Communication latency.
- Gravity and payload effects.

Use reduced-speed execution initially.

Any time scaling or trajectory interpolation must be validated before physical execution.

The trajectory action server should compare commanded positions with measured joint positions during execution.

Abort or stop when tracking errors exceed configured limits.

### 17.2 Pre-execution verification

Before accepting or dispatching an outbound trajectory, verify:

- Robot connection.
- Current operational state.
- Measured joint position validity.
- Agreement between the measured starting configuration and the trajectory's initial configuration.
- Obstacle measurement validity.
- Execution authorization.
- No active hardware faults.

If the robot has moved significantly since planning, invalidate the trajectory.

If the obstacle has changed significantly since planning, invalidate the trajectory.

### 17.3 Execution completion

After reaching the target:

- Verify the final measured joint configuration.
- Record the execution result.
- Calculate estimated end-effector position using forward kinematics.
- Transition the robot into AT_TARGET.
- Store the outbound trajectory for potential return planning.

Do not immediately mark the robot as READY.

## 18. Return trajectory planning

The PPO policy was trained to reach targets from the original starting configuration.

Do not use it directly for arbitrary return-to-home motions unless independently demonstrated to work.

### 18.1 Reverse trajectory candidate

Initially investigate reversing the selected outbound joint-position trajectory.

The planner should:

1. Read the current measured joint configuration.
2. Verify that the obstacle has not moved.
3. Retrieve the previously executed outbound trajectory.
4. Reverse the joint-position sequence.
5. Generate appropriate new timestamps.
6. Account for differences between the expected and measured final configuration.
7. Validate the entire return trajectory in MuJoCo.
8. Check collisions, clearance, joint limits and velocity limits.
9. Send the return trajectory only if it passes validation.

Do not reverse the original PPO actions.

Do not assume that simply reversing actuator commands reproduces the outbound path.

### 18.2 Return completion

After executing a return trajectory:

- Verify measured home joint positions.
- Verify gripper configuration.
- Wait for the robot to settle.
- Verify stationary state.
- Transition to READY.

The camera may then acquire a new obstacle image.

### 18.3 Return failure

If a valid return trajectory cannot be generated:

- Keep the robot stationary.
- Report the reason.
- Allow the operator to clear the workspace.
- Perform the standard homing procedure only after the workspace is confirmed safe.

Do not introduce another reinforcement learning policy solely for returning home unless necessary.

## 19. Safety requirements

Simulation-based validation reduces risk but does not guarantee physical safety.

Independent safeguards are required.

### 19.1 Hardware safeguards

Implement:

- Joint position limits.
- Joint velocity limits.
- Trajectory tracking error limits.
- Communication watchdog.
- Trajectory cancellation.
- Hardware error handling.
- Appropriate controlled stopping behavior.
- Accessible physical emergency stop.
- Conservative execution speed.

### 19.2 Planning safeguards

Implement:

- Obstacle measurement validity checks.
- Joint state freshness checks.
- Starting configuration verification.
- Collision checking.
- Minimum clearance validation.
- Planning timeout.
- Input validation.
- Invalid trajectory rejection.
- Return trajectory validation.

### 19.3 Execution authorization

Physical execution must be disabled by default.

Use explicit configuration options such as:

```yaml
use_mock_hardware: true
enable_hardware_execution: false
require_operator_confirmation: true
auto_return_home: false
```

The exact parameter names can be adjusted.

Initially require operator confirmation before executing a planned trajectory.

Autonomous execution may be enabled after physical validation.

### 19.4 Environment changes

A planned trajectory is only valid for the robot and obstacle configuration used during planning.

If the robot moves unexpectedly or the obstacle changes position:

- Invalidate the trajectory.
- Stop or cancel execution where appropriate.
- Replan only after obtaining valid measurements.

The initial demonstration assumes the wooden stick remains static during outbound and return execution.

If continuous obstacle monitoring is unavailable, document this limitation and maintain a controlled workspace.

Do not describe the system as capable of safely avoiding arbitrary moving obstacles.

A software stop service must not be presented as a replacement for an independent physical emergency stop.

## 20. Configuration files

Use YAML configuration files rather than hardcoding hardware or planning settings.

### 20.1 Robot configuration

Include:

- Hardware interface type.
- Serial port.
- Baud rate.
- Motor IDs.
- Joint names.
- Servo calibration.
- Motor direction.
- Joint limits.
- Velocity limits.
- Tracking tolerances.
- Feedback frequency.
- Communication timeout.
- Home configuration reference.
- Home position tolerances.
- Homing speed.
- Settling requirements.
- Hardware execution enable flag.

The home configuration reference must resolve to the original training starting configuration.

Do not maintain a second independently defined set of home joint angles.

### 20.2 Camera configuration

Include:

- Camera device.
- Image resolution.
- Frame rate.
- Calibration file.
- Camera intrinsics.
- Camera mounting transform.
- Table-plane parameters.
- Known obstacle dimensions where applicable.
- Detection thresholds.
- Minimum detection confidence.
- Maximum detection age.
- Requirement for verified home configuration.

### 20.3 Planner configuration

Include:

- MuJoCo model path.
- PPO checkpoint path.
- Inference device.
- Candidate target count.
- Minimum target distance.
- Maximum target distance.
- Target offsets around obstacle.
- Target height restrictions.
- Minimum starting clearance.
- Required trajectory clearance.
- Maximum rollout steps.
- Planning timeout.
- Deterministic seed.
- Starting configuration tolerance.
- Return planning parameters.

Values not established by existing code or measurements must be clearly marked as provisional.

## 21. ROS 2 development environment

The existing RL training environment has been used on Windows.

Before implementation, determine the intended ROS 2 deployment environment.

Consider compatibility between:

- ROS 2.
- Python.
- MuJoCo.
- Gymnasium.
- Stable-Baselines3.
- PyTorch.
- OpenCV.
- Hiwonder motor communication libraries.
- USB and serial device access.

Do not assume that the existing Windows Python environment can directly run every ROS 2 dependency.

Document the chosen development and deployment environment.

If using WSL2, Docker or Linux, verify physical camera and serial-device access before committing to the environment.

Provide reproducible installation instructions.

Avoid unnecessary containerization or infrastructure complexity.

## 22. Implementation milestones

Implement the project incrementally.

Do not attempt to complete all phases simultaneously.

### Phase 0 — Repository audit

Inspect the existing codebase.

Identify:

- Environment structure.
- PPO observation and action spaces.
- Exact training starting configuration.
- Gripper configuration.
- MuJoCo joint names.
- Robot control timing.
- Existing camera collision geometry.
- Existing Hiwonder hardware code.
- Existing calibration files.
- Existing dependencies.
- Available checkpoints.
- Coordinate frame conventions.

Identify missing hardware information.

Create `IMPLEMENTATION_PLAN.md` with a progress checklist.

Document all assumptions.

Acceptance criteria:

- Existing architecture is understood.
- Observation and action compatibility is verified.
- Original training home configuration is identified.
- Required hardware information is identified.
- No unsupported calibration or motor specifications are introduced.

### Phase 1 — ROS 2 foundation

Implement:

- ROS 2 workspace.
- Application package.
- Custom interface package.
- Three application node skeletons.
- Topics.
- Services.
- Trajectory execution action.
- Homing action.
- Robot operational states.
- Launch files.
- YAML configuration files.
- Shared home configuration access.
- Mock hardware interface.
- Mock obstacle input.

Acceptance criteria:

- All three application nodes launch successfully.
- Nodes exchange messages.
- Mock joint feedback is published.
- Mock obstacle data is received.
- Mock homing functionality works.
- Robot state correctly transitions to READY after homing.
- No physical hardware is required.
- No physical movement is enabled by default.

### Phase 2 — Standalone PPO planner

Implement:

- Explicit MuJoCo initialization.
- Original home configuration verification.
- Existing PPO checkpoint loading.
- Candidate target generation.
- Deterministic policy rollouts.
- Trajectory recording.
- Collision checking.
- Minimum clearance calculation.
- Trajectory rejection.
- Trajectory ranking.
- JointTrajectory generation.
- Reverse trajectory generation and validation.

Initially use manually supplied obstacle coordinates and mock joint states.

Acceptance criteria:

- Planning works without the physical robot.
- Planning works without the physical camera.
- Planning is rejected if the robot is not at home.
- Explicit obstacle coordinates are respected.
- Explicit target coordinates are respected.
- Deterministic inputs produce reproducible results.
- Successful trajectories can be selected.
- Colliding trajectories are rejected.
- No execution goal is sent when all candidates fail.
- Return trajectories are independently validated.
- Existing training and evaluation scripts remain functional.

This phase is the highest initial implementation priority.

### Phase 3 — Hiwonder robot integration

Implement:

- Hiwonder communication interface.
- Motor discovery where supported.
- Motor-to-joint mapping.
- Servo calibration.
- Measured joint feedback.
- Unit conversions.
- Physical homing functionality.
- Home configuration verification.
- Trajectory execution action server.
- Tracking monitoring.
- Cancellation.
- Communication watchdog.
- Controlled stopping.

Begin with individual joint movements.

Then test synchronized multi-joint movements and homing without obstacles.

Acceptance criteria:

- Correct motor IDs are identified.
- Physical joint positions agree with simulation conventions.
- Joint directions and zero offsets are verified.
- The physical robot can reproduce the original training starting configuration.
- The robot can perform repeated homing operations.
- Homing accuracy and repeatability are measured.
- The robot can reproduce simple simulated trajectories.
- Physical tracking error is measured.
- Cancellation functions correctly.
- Communication failures are handled safely.
- Physical execution remains explicitly configurable.

### Phase 4 — Camera integration

Implement:

- Camera capture.
- Camera calibration loading.
- Image publication.
- Wooden stick detection.
- Metric spatial mapping.
- Camera-to-base transformations.
- Obstacle message publication.
- Detection validity checks.
- Image acquisition from verified home configuration.

Test obstacle mapping against measured physical reference positions.

Acceptance criteria:

- The wooden stick is detected reliably under demo conditions.
- Planning images are acquired while the robot is stationary at home.
- Mapped obstacle coordinates are expressed in the robot base frame.
- Mapping error is measured.
- Repeated homing operations produce consistent obstacle mappings.
- Invalid detections are rejected.
- Stale detections are rejected.

### Phase 5 — Full system integration

Connect the camera, planner and robot nodes.

Implement the complete sequence:

1. Request homing.
2. Verify the original training starting configuration.
3. Obtain valid measured joint positions.
4. Obtain a valid obstacle map.
5. Generate candidate targets.
6. Simulate PPO trajectories.
7. Reject unsafe or unsuccessful trajectories.
8. Rank successful trajectories.
9. Publish the selected target and trajectory.
10. Obtain execution authorization.
11. Execute the selected trajectory.
12. Monitor physical execution.
13. Report success or failure.
14. Generate and validate the return trajectory.
15. Execute the return trajectory.
16. Verify the home configuration.
17. Return to READY.

Acceptance criteria:

- Moving the wooden stick generates a different obstacle map.
- The system automatically generates a new target.
- The PPO policy generates the motion.
- MuJoCo validates the motion.
- Physical execution only occurs for an accepted validated trajectory.
- Invalid configurations result in no movement.
- The robot can return to its original training starting configuration.
- The complete demonstration can be repeated.

### Phase 6 — Visualization and experimental evaluation

Add RViz visualization.

Display:

- Robot configuration.
- Robot coordinate frames.
- Camera frame.
- Original home configuration where useful.
- Mapped wooden stick.
- Conservative obstacle collision geometry.
- Candidate targets.
- Rejected targets.
- Selected target.
- Planned end-effector trajectory.
- Return trajectory.

Use standard robot visualization infrastructure where practical.

Create a URDF representation if required for RViz, keeping its kinematic conventions consistent with MuJoCo.

Do not introduce MoveIt 2 unless it becomes necessary.

Acceptance criteria:

- The operator can inspect the mapped obstacle.
- The operator can inspect candidate targets.
- The selected trajectory is visible before execution.
- The return trajectory can be inspected.
- Planning failures have understandable diagnostic messages.
- Demonstration results can be recorded and evaluated.

## 23. Testing strategy

### 23.1 Unit tests

Provide tests for:

- Servo-unit-to-radian conversion.
- Radian-to-servo-unit conversion.
- Motor direction handling.
- Joint offset calibration.
- Home configuration consistency.
- Home tolerance checking.
- Robot state transitions.
- Coordinate transformations.
- Observation construction.
- Candidate target generation.
- Candidate reachability filtering.
- Trajectory ranking.
- Return trajectory generation.
- Invalid input handling.
- Stale measurement detection.
- Joint limit enforcement.

### 23.2 Simulation integration tests

Test:

- Explicit MuJoCo initialization.
- Original training starting configuration.
- PPO checkpoint compatibility.
- Deterministic inference.
- Reproducible trajectories.
- Collision rejection.
- Minimum clearance calculation.
- Invalid starting configurations.
- All-candidates-failed scenarios.
- Reverse trajectory validation.
- Existing randomized environment resets.

### 23.3 ROS integration tests

Test using mock hardware:

- Node startup.
- Topic communication.
- Service communication.
- Homing action acceptance.
- Homing cancellation.
- Robot state publication.
- READY state verification.
- Planning rejection when the robot is not at home.
- Trajectory action acceptance.
- Action cancellation.
- Execution feedback.
- Planning failure behavior.
- Return-to-home behavior.
- Tracking error handling.
- Communication watchdog behavior.

### 23.4 Physical tests

Progress gradually:

1. Individual joint movement.
2. Joint position feedback.
3. Synchronized joint movement.
4. Homing in a clear workspace.
5. Repeated home positioning accuracy.
6. Obstacle-free trajectory execution.
7. Obstacle-free return-to-home execution.
8. Camera mapping validation.
9. Simulated planning using mapped physical obstacles.
10. Reduced-speed physical obstacle avoidance.
11. Validated obstacle-aware return execution.
12. Complete demonstration.

Do not begin physical obstacle avoidance testing before validating joint conventions, homing accuracy and physical trajectory tracking.

## 24. Experimental evaluation

Record useful quantitative measurements.

### 24.1 Homing metrics

- Home position error per joint.
- Maximum joint error.
- Homing repeatability.
- Homing execution duration.
- Homing success rate.
- Camera mapping consistency after repeated homing.

### 24.2 Perception metrics

- Obstacle detection success rate.
- Obstacle position mapping error.
- Obstacle dimension estimation error.
- Mapping repeatability.
- Detection processing time.

### 24.3 Planning metrics

- Number of generated candidates.
- Number of reachable candidates.
- Number of successful PPO rollouts.
- Number of rejected trajectories.
- Collision rejection rate.
- Planning computation time.
- Selected trajectory minimum clearance.
- Selected trajectory duration.
- Return trajectory validation success rate.

### 24.4 Hardware metrics

- Joint tracking error.
- Maximum tracking error.
- Final end-effector error where measurable.
- Physical execution duration.
- Physical reaching success rate.
- Physical return success rate.
- Aborted execution rate.
- Physical collision rate.

### 24.5 System-level metrics

- Percentage of obstacle configurations for which at least one valid candidate is found.
- Percentage of planning requests resulting in successful physical execution.
- Percentage of complete outbound-and-return demonstration cycles.
- End-to-end processing time.
- Failure reasons.

Distinguish between:

1. Individual PPO trajectory success rate.
2. Candidate-based planning success rate.
3. Physical execution success rate.
4. Complete demonstration cycle success rate.

These are different performance measurements.

Compare the 590k and 635k checkpoints on identical evaluation seeds if both checkpoints are available.

Do not select the final model solely based on evaluations using different seed distributions.

## 25. Logging and diagnostics

Use ROS logging for runtime information.

Provide meaningful messages such as:

```text
[Robot] Connected to Hiwonder motors
[Robot] Moving to training home configuration
[Robot] Home configuration verified
[Robot] State: READY
[Camera] Valid obstacle detected
[Camera] Obstacle mapped to base_link
[Planner] Generated 15 candidate targets
[Planner] Candidate 1 rejected: camera collision
[Planner] Candidate 2 rejected: unreachable target
[Planner] Candidate 3 accepted
[Planner] Selected trajectory with highest clearance
[Robot] Trajectory goal accepted
[Robot] Executing trajectory
[Robot] Execution completed
[Robot] State: AT_TARGET
[Planner] Return trajectory validated
[Robot] Returning to home
[Robot] Home configuration verified
[Robot] State: READY
```

Record planning results in a structured format such as JSON or CSV.

Include:

- Obstacle geometry.
- Robot starting configuration.
- Home position verification.
- Candidate target positions.
- Candidate rejection reasons.
- Selected trajectory statistics.
- Execution outcome.
- Return trajectory outcome.

Where useful, record ROS topics using rosbag for debugging and experimental analysis.

Avoid excessive logging inside high-frequency control loops.

## 26. Final demonstration procedure

The intended demonstration should be simple and repeatable.

### Initial setup

1. Launch the three ROS 2 application nodes.
2. Establish communication with the Hiwonder motors and camera.
3. Confirm the workspace is clear.
4. Request homing.
5. Move the robot into its original training starting configuration.
6. Verify the measured joint positions.
7. Wait for the robot to settle.
8. Confirm the robot is READY.

### Demonstration

1. Place the wooden stick in the defined robot workspace.
2. Trigger the demonstration.
3. Capture a camera image from the home configuration.
4. Detect the wooden stick.
5. Map its geometry into robot base coordinates.
6. Generate candidate reaching targets near the stick.
7. Run the trained PPO policy in MuJoCo for each candidate.
8. Reject colliding and unsuccessful trajectories.
9. Select the successful trajectory with the greatest acceptable clearance.
10. Visualize the mapped obstacle, selected target and planned trajectory.
11. Authorize execution.
12. Execute the trajectory using the Hiwonder motors.
13. Report physical execution success or failure.

### Return and repeat

1. Request return-to-home.
2. Obtain the current measured robot configuration.
3. Generate a candidate return trajectory.
4. Validate the return trajectory against the unchanged obstacle.
5. Execute the validated return trajectory.
6. Verify the original training starting configuration.
7. Confirm the robot is stationary and READY.
8. Reposition the wooden stick.
9. Begin the next demonstration.

If the return trajectory cannot be validated, keep the robot stationary and request operator intervention.

### Proposed commands

Launch:

```bash
ros2 launch so101_demo demo.launch.py
```

Move into the original training starting configuration:

```bash
ros2 action send_goal /robot/move_to_home so101_interfaces/action/MoveToHome "{}"
```

Start obstacle detection and planning:

```bash
ros2 service call /start_demo std_srvs/srv/Trigger "{}"
```

Authorize execution:

```bash
ros2 service call /execute_plan std_srvs/srv/Trigger "{}"
```

Return to home:

```bash
ros2 service call /return_home std_srvs/srv/Trigger "{}"
```

Request a controlled stop:

```bash
ros2 service call /stop_robot std_srvs/srv/Trigger "{}"
```

These interfaces must be implemented and tested before they are considered functional.

## 27. Development constraints

- Maintain exactly three primary application nodes.
- Use ROS 2 and Python.
- Preserve the existing reinforcement learning repository.
- Reuse the existing MuJoCo model.
- Reuse the trained PPO policy.
- Preserve the policy observation and action spaces.
- Preserve existing action conversion logic.
- Preserve camera collision geometry.
- Use the exact original training starting configuration.
- Maintain one authoritative home configuration.
- Implement dedicated robot homing functionality.
- Verify home positioning using measured motor feedback.
- Acquire planning images while the robot is stationary at home.
- Validate return-to-home trajectories independently.
- Use Hiwonder motor communication rather than assuming Feetech compatibility.
- Do not invent hardware specifications.
- Do not invent calibration values.
- Do not retrain PPO unless explicitly requested.
- Do not introduce MoveIt 2 initially.
- Do not introduce unnecessary perception models.
- Do not manually program outbound reaching waypoints for the final demonstration.
- Do not hardcode final obstacle or target positions.
- Do not directly execute unvalidated PPO actions.
- Disable physical execution by default.
- Preserve simulation-only operation.
- Keep changes incremental, documented and testable.
- Prioritize reliability over unnecessary complexity.

## 28. Instructions for Codex

Begin by inspecting the existing repository.

Do not immediately rewrite or reorganize the entire codebase.

First:

1. Audit the existing RL environment and policy.
2. Verify observation and action compatibility.
3. Extract the exact original training starting configuration.
4. Inspect existing Hiwonder motor communication code.
5. Inspect camera geometry and calibration information.
6. Determine how existing simulation modules can be reused by ROS 2.
7. Identify missing hardware information.
8. Identify dependency and operating system constraints.
9. Create `IMPLEMENTATION_PLAN.md`.

Then implement Phase 1 and Phase 2.

The initial deliverable should be a working ROS 2 application with:

- Three application nodes.
- Mock hardware.
- Shared original home configuration.
- Mock homing functionality.
- Robot operational states.
- Manually supplied obstacle coordinates.
- Automatic target generation.
- PPO trajectory simulation.
- Collision-based trajectory selection.
- Validated return trajectory generation.

Do not begin physical hardware integration until the simulation-based planner is working correctly.

After each implementation phase:

- Run available tests.
- Report modified and created files.
- Document implementation decisions.
- Document test results.
- Document known limitations.
- Update the implementation checklist.
- Identify remaining work.

If hardware-specific information is missing, document it and create an appropriate configurable interface rather than inventing values.

Do not claim that physical execution, homing accuracy or calibration has been validated unless actual physical testing has been performed.

## 29. Definition of project completion

The project is complete when the physical SO-ARM101, equipped with Hiwonder motors and a wrist-mounted camera, can:

- Move into the exact original training starting configuration.
- Verify home positioning using measured joint feedback.
- Detect a wooden stick from the home configuration.
- Map the obstacle into robot base coordinates.
- Automatically generate nearby reaching targets.
- Generate candidate motions using the existing PPO policy.
- Validate those motions in MuJoCo.
- Select a successful trajectory with sufficient clearance.
- Execute that trajectory using the Hiwonder motors.
- Monitor physical execution.
- Reject unsafe configurations.
- Generate and execute a validated return trajectory.
- Return to the original training starting configuration.
- Repeat the process when the obstacle is repositioned.

The final contribution is a modular vision-guided robotic system that combines reinforcement learning with simulation-based trajectory validation to perform autonomous reaching near obstacles within a defined operating envelope.
