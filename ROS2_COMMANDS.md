# ROS 2 command reference

This guide records the ROS 2 commands for this repository's `ros2_ws` workspace.
The workspace provides a `robot_node` with mock and opt-in physical modes and
a `planner_node` that simulates reaches from measured joints and a manually supplied obstacle. Build the
workspace to generate the Python interfaces.

## Environment

The development environment used to prepare this guide exposes ROS 2 Jazzy and
`colcon`. Check the active environment before building:

```bash
printenv ROS_DISTRO
command -v ros2
command -v colcon
```

On a Jazzy installation, source the ROS underlay in each new Bash terminal, then
source the workspace overlay after building:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
```

If using another ROS 2 distribution, replace `jazzy` with its installed
distribution in the source path. Do not mix distributions in one terminal.

## Install dependencies and build

From the repository root, after sourcing the underlay:

```bash
cd ros2_ws
rosdep install --from-paths src --ignore-src --rosdistro "$ROS_DISTRO" -r -y
colcon build --symlink-install
source install/setup.bash
```

After changing source files, rebuild and source the overlay again:

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

To build only one package and its dependencies:

```bash
colcon build --symlink-install --packages-up-to so101_demo
```

`rosdep` requires network access and system package installation rights. If
dependencies are already installed, the build and overlay commands are enough.

## Read the physical follower in one terminal

Run from the repository root with the follower's USB serial adapter visible
in Linux. Use the same Hiwonder checkout and **follower** calibration that
were used to calibrate this arm. Replace the three local values below.

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
export HIWONDER_FORK='<local Hiwonder fork directory>'
export FOLLOWER_CALIBRATION='<local follower calibration JSON>'
.venv/bin/python -m serial.tools.list_ports -v
export ARM_PORT='<listed follower serial device>'
export PYTHONPATH="$HIWONDER_FORK/src:$PYTHONPATH"
.venv/bin/python -c 'import inspect; from lerobot.motors.hiwonder import HiwonderMotorsBus; print(inspect.getfile(HiwonderMotorsBus))'
ros2 run --prefix "$PWD/.venv/bin/python" so101_demo robot_node \
  --read-only --once --port "$ARM_PORT" --calibration "$FOLLOWER_CALIBRATION"
```

The import check should print the bus module from the Hiwonder checkout.
For the inspected fork at revision `a24998f7`, the reader can use the
checkout directly with NumPy, pySerial, DeepDiff and tqdm on the Linux
interpreter's import path. In the project simulation environment, install
missing reader dependencies with
`.venv/bin/python -m pip install 'pyserial>=3.5,<4' 'deepdiff>=7.0.1,<9' tqdm`.
The full fork installation is described in the README.

The check reads the programmed homing offset and both position limits at
each configured ID, compares them with the calibration file and prints one
set of six raw encoder counts. It uses the fork's default host baud rate;
`--baudrate` selects a known alternative. An unsuccessful read ends the
reader and reports the error. Closing the port preserves the arm's existing
torque state. Omit `--once` to watch the counts. Keep calibration and
reference measurements in local files outside the repository.

### Jog the follower wrist

Use the same terminal, follower calibration and Python environment as the
successful encoder read. Close other programs using the port, support the
arm, and leave clearance around the wrist and its camera cable.

```bash
ros2 run --prefix "$PWD/.venv/bin/python" so101_demo robot_node \
  --jog-wrist --port "$ARM_PORT" --calibration "$FOLLOWER_CALIBRATION"
```

Each run requests one positive two-degree change in wrist encoder coordinates.
The wrist must start stationary in position mode with torque off. The command
uses the current count as the hold target. Writing this target can engage wrist
torque. The command verifies the target, torque, position and stationary feedback
throughout the startup hold, advances the target in small steps, and prints the measured displacement
after verifying torque is off again. If serial communication fails while
powered, a torque-off command may also fail. Keep the power switch accessible.

### Align encoder counts with ROS joint angles

Use the printed raw counts to verify which physical joint corresponds to
each name. For each joint, including the gripper, record its count at the
independently measured training-home angle. Then record its count and
independently measured angle at a second pose on the same single-turn encoder
interval. Angles follow the MuJoCo joint axes in radians. The two poses must
differ by more than twice the chosen measurement tolerance. This procedure
assumes a direct drive and a continuous count interval throughout the
intended motion. Check for encoder wrap before using that interval.

Create a local JSON file with three top-level fields:

- `calibration_sha256` is the digest printed by the raw check. A changed
  calibration requires rechecking the mapping.
- `tolerance_rad` is the positive angular uncertainty of the reference measurement.
- `joints` is an object keyed by all six names from `training_home.json`.
  Each entry contains integer `home_tick` and `check_tick` and numeric
  `check_rad`, the absolute joint angle at the second pose.

The reader takes scale from the HX-30HM driver table, checks it against the
second measurement and determines the sign for each axis. The programmed
homing offset is already reflected in `Present_Position`. To inspect both
ticks and radians before streaming, add `--mapping <local-mapping.json>`
to the one-shot command. Keep the mapping argument and omit `--once` to
publish timestamped, complete position samples on `/joint_states`. The
velocity and effort arrays are empty. Physical feedback mode exposes no
motion actions and leaves the existing mock workflow accessible through the
default invocation.

For a follower observed in the model's all-zero calibration pose, the mapping
also accepts `"reference_pose": "model_zero"`, the same calibration digest,
and a `joints` object containing each joint's captured `zero_tick` and measured
`sign` (1 or -1). In this format the angle is the signed tick displacement
from that zero times the encoder scale. Training home still comes from
`training_home.json`. Physical accuracy depends on the pose used for capture.

### Six-joint physical controller

After the follower mapping and physical limits have been verified, start the
controller in the sourced ROS terminal from the repository root:

```bash
ros2 run --prefix "$PWD/.venv/bin/python" so101_demo robot_node \
  --hardware-control --port "$ARM_PORT" --calibration "$FOLLOWER_CALIBRATION" \
  --mapping "$JOINT_MAPPING" --motion-config "$MOTION_CONFIG"
```

`JOINT_MAPPING` is the local two-reference file described above. `MOTION_CONFIG`
is a local JSON object with the following fields. Set values from measurements
of the assembled follower, including the gripper and mounted camera.

| Field | Meaning |
| --- | --- |
| `joints` | Object keyed by each of the six training joint names. Each entry contains `min_rad`, `max_rad`, and `max_rad_s`. Bounds must include home and lie within the mapped encoder interval. |
| `home_tolerance_rad` | Maximum per-joint error at the start of a path and at a settled endpoint. |
| `tracking_error_rad` | Maximum per-joint deviation for initial hold, endpoint hold, and trajectory execution outside Home. |
| `stationary_rad_s` | Maximum absolute measured angular speed for a stationary joint. The servo's moving flag must also be clear. |
| `feedback_timeout_s` | Feedback watchdog deadline. The physical controller requests a six-servo snapshot every 5 ms and logs gaps of 10 ms or longer. Home sends up to 30 position commands per second. Other trajectories space commands at least 50 ms apart. |
| `settle_timeout_s` | Time allowed to verify the initial powered hold or final settled pose. Each requires 0.1 s of stationary feedback within position tolerance. |
| `home_path` | Optional intermediate poses and minimum segment times. Each waypoint contains `time_s` and six `positions_rad` in training order. Time starts at zero and the last waypoint is training home. Every Home request replaces the first pose with fresh joint measurements. Omit the path for a direct approach. |

Startup publishes measured `/joint_states` and `/robot/state`. An accepted
`/robot/move_to_home` action with `workspace_clear: true` authorizes the first
powered hold. Each Home request checks the approach against joint and speed
limits and samples it for collisions in the clear-workspace MuJoCo model,
including the wrist camera and floor. A collision rejects the request.
Home reads the follower position and uses the same six-joint movement
function as `scripts/hardware/check_arm_motion.py`. It advances each frame and retries
the segment endpoint for up to three seconds until all arm axes are within one
degree and the gripper is within one point on its 0–100 scale. ROS then verifies
that the arm is stationary within the configured training-home tolerance.
Segment timing accounts for the peak commanded speed midway through the move.
During Home and physical policy replay, measured speeds are reported on
`/joint_states`. A speed spike can delay settling. The command-speed limit
still applies to every segment. The ordinary `/execute_trajectory` action
also enforces a measured-speed cutoff.
From READY, `/execute_trajectory` accepts an immediately starting
`control_msgs/action/FollowJointTrajectory` with all six names in training
order and timed positions. Supply a collision-validated path. The server checks
its start, bounds and segment speeds. From AT_TARGET, a subsequent trajectory
must end at home.

Cancel the action or call `/stop_robot` (`std_srvs/srv/Trigger`) to stop issuing
trajectory points. On stop, fault, or shutdown the controller attempts to hold
each reachable powered axis at its measured position so the arm stays supported.
A bus or power failure can prevent that hold. A controller fault requires a
restart after inspecting the arm. Keep the independent physical stop accessible.

## Replay the Stage 1 obstacle policy on the follower

Use the Linux project environment, a common ROS domain, and the repository root
in each terminal. The physical controller needs dialout access and the exported
`FOLLOWER_CALIBRATION` path. Start one controller for the session. It reads
the encoders while waiting for an execution request.

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
export PYTHONPATH="$PWD/scripts:$PYTHONPATH"
```

In the first terminal run the controller:

```bash
ros2 run --prefix "$PWD/.venv/bin/python" so101_demo robot_node \
  --hardware-control --port /dev/ttyACM0 --calibration "$FOLLOWER_CALIBRATION" \
  --mapping outputs/follower-zero.json --motion-config outputs/follower-home.json \
  --ros-args -p physical_replay:=true
```

In a second terminal run the planner:

```bash
ros2 run --prefix "$PWD/.venv/bin/python" so101_demo planner_node \
  --ros-args -p physical_replay:=true -p motion_config:=outputs/follower-home.json
```

Supply the measured center of the fixed training block on `/obstacle_center`
in world-frame meters. The model uses an upright 36 × 36 × 240 mm block
centered at z = 0.13 m. For manual input, enter its measured X and Y in a third terminal:

```bash
read -rp 'Measured block X Y in meters: ' BLOCK_X BLOCK_Y
ros2 topic pub --once -w 2 /obstacle_center geometry_msgs/msg/PointStamped \
  "{header: {stamp: now, frame_id: world}, point: {x: $BLOCK_X, y: $BLOCK_Y, z: 0.13}}"
```

This publishes once after both nodes subscribe. Each node saves the coordinates,
so you can use the same terminal for the service calls. If you reposition the
block, enter the new center and request a new plan. A changed measurement
invalidates a waiting plan and stops an active replay. If the block moves
during a reach, call `/stop_robot`. Enter the center again after restarting
either node.

Monitor `ros2 topic echo /planner/status std_msgs/msg/String` in another
terminal. Request planning from the stationary measured pose:

```bash
ros2 service call /start_demo std_srvs/srv/Trigger '{}'
```

Physical replay accepts a stationary IDLE or READY start within 5° of training
home at elbow flex, 1.5° at wrist flex, and 1° at each remaining joint. MuJoCo starts at those measured
angles. It evaluates candidate policy reaches with physical actuator limits
and publishes the selected simulated **joint positions** on `/planned_trajectory`.
It stretches segment times to fit the physical speed limits and checks the
linear interpolation against the model and the configured block clearance
(`policy_clearance_m` in the shared motion config, default 0.002 m).

After `PLAN_READY`, authorize one physical reach:

```bash
ros2 service call /execute_plan std_srvs/srv/Trigger '{}'
```

The planner verifies fresh joint and robot-state messages before dispatch.
After dispatch it waits for the action result. The controller monitors encoder
feedback, obstacle clearance, and stop requests. It reads all six encoders directly
and checks that the measured pose matches the first waypoint within
`home_tolerance_rad`. It checks again when the initial hold settles.
The controller streams the retimed waypoints through the Hiwonder follower,
allows joints to lag during playback, and checks the segment from the measured
pose to each capped command for obstacle clearance. It reports `AT_TARGET`
once the endpoint settles. If settling times out, it holds the measured pose
and reports each joint's signed endpoint error in degrees (measured minus planned).
During replay the operator can stop with
`ros2 service call /stop_robot std_srvs/srv/Trigger '{}'`.
Physical policy replay currently covers the outbound reach.

## Run the current scaffold

Run the mock robot with the overlay sourced:

```bash
ros2 run so101_demo robot_node
```

In another sourced terminal, start the planner from the **repository root**.
The Jazzy console script uses system Python. Select the project interpreter
with the simulation dependencies and expose the standalone planner on the
import path:

```bash
PYTHONPATH="$PWD/scripts:$PYTHONPATH" ros2 run --prefix "$PWD/.venv/bin/python" so101_demo planner_node
```

From a third sourced terminal, query the current start conditions:

```bash
ros2 service call /planner/check_start std_srvs/srv/Trigger '{}'
```

The response gives the first failed check or confirms that the measured
start is ready for planning. Each call checks both topics again. The default
`feedback_timeout_sec` is 0.5 s for both sample age and time since receipt.
`home_tolerance_rad` and `stationary_speed_rad_s` default to 0.01 rad and
0.01 rad/s per joint. This service only reports readiness.

The mock robot source publishes `/joint_states` and `/robot/state` at 20 Hz.
Its `/robot/move_to_home` goal takes a boolean `workspace_clear` field.
Set this to true for the initial approach in the empty mock workspace.
The package test `test_mock_homing.py` exercises the action, feedback,
cancellation, subsequent homing and READY through the ROS interfaces.

There is no `demo.launch.py` in the current package, so the launch command in
`PROJECT_SPECS.md` is a proposed future interface, not a working command yet.

## Run a mock round trip in one terminal

From the repository root, source Jazzy and the built overlay. Run this
integration test to home the mock robot, plan a real PPO reach, explicitly
request execution, then request a validated return to stationary training
home. The test offsets the mock endpoint measurement by 0.003 rad before
the return to verify the planner uses the live feedback:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
PYTHONPATH="$PWD/scripts:$PYTHONPATH" .venv/bin/python -m pytest -q \
  'ros2_ws/src/so101_demo/test/test_mock_execution.py::test_mock_execution[round_trip]'
```

The test starts the ROS nodes in its own process. It requires the local
Stage 1 checkpoint and Jazzy's `control_msgs` package. Rebuild the workspace
after updating `RobotState.msg` so it includes `EXECUTING` and `AT_TARGET`.
Use the `reaches_endpoint` case for outbound motion only.
To also exercise obstacle changes before, during and after the reach, pass the whole
`test_mock_execution.py` file to pytest.

## Plan a mock reach

With the mock workspace clear, home the mock robot and wait for success:

```bash
ros2 action send_goal /robot/move_to_home so101_interfaces/action/MoveToHome '{workspace_clear: true}'
```

Publish the training block center at 10 Hz in another sourced terminal.
The example uses MuJoCo world coordinates in meters. The simulator supplies
the block's fixed orientation and dimensions and requires a center height of
0.13 m. Keep the publisher running until the search finishes:

```bash
ros2 topic pub -r 10 /mock/obstacle_center geometry_msgs/msg/PointStamped \
  '{header: {stamp: now, frame_id: world}, point: {x: 0.29336519, y: -0.09268185, z: 0.13}}'
```

Start each inspection command in its own sourced terminal before requesting
the plan. The trajectory topic publishes once per successful search:

```bash
ros2 topic echo /planner/status std_msgs/msg/String
```

```bash
ros2 topic echo /planned_trajectory trajectory_msgs/msg/JointTrajectory --once
```

Request the reach:

```bash
ros2 service call /start_demo std_srvs/srv/Trigger '{}'
```

A successful service response acknowledges the search. The status topic
then reports `PLAN_READY` with the target and minimum clearance, or `FAILED`
with a reason. `/planned_trajectory` contains the six simulated actuator
targets in radians and their times from the measured start. This step only
publishes the plan for inspection. The default `min_clearance_m` of 0.002 m
is a simulation setting. Set `checkpoint_path` at startup to override the
standalone planner's default Stage 1 checkpoint.

To authorize this reach on the mock robot, keep the obstacle publisher
running and call:

```bash
ros2 service call /execute_plan std_srvs/srv/Trigger '{}'
```

The service rechecks the measured start and obstacle. An accepted request
consumes the plan and dispatches the position commands to the mock
`FollowJointTrajectory` action at `/mock/execute_trajectory`. Monitor
`/planner/status` for `EXECUTING` and then `AT_TARGET` or `FAILED`.
The ideal mock follower interpolates the commanded joint positions over
their published times. Repositioning the block invalidates a waiting plan
and requests cancellation of an active goal.

After `AT_TARGET`, keep the block fixed and its publisher running. Request
the return separately:

```bash
ros2 service call /return_home std_srvs/srv/Trigger '{}'
```

The request authorizes one return if MuJoCo validates it from the current
joint positions and velocities. Monitor `/planner/status` for
`VALIDATING_RETURN`, `RETURN_VALIDATED`, `RETURNING` and finally `READY`.
A failed validation leaves the mock at the endpoint. An observed block
movement invalidates the return even if the block moves back. The mock
reports `READY` after all six joints settle at home for 0.1 s.

## Common ROS 2 inspection commands

These commands are useful once one or more actual nodes are running. Run them
from another terminal with both the Jazzy underlay and workspace overlay
sourced.

```bash
ros2 node list
ros2 topic list -t
ros2 service list -t
ros2 action list -t
ros2 interface list
```

Inspect a specific running node, topic, or interface:

```bash
ros2 node info /node_name
ros2 topic info /topic_name --verbose
ros2 interface show package_name/msg/MessageName
```

For live data, use the type printed by `ros2 topic list -t`:

```bash
ros2 topic echo /topic_name
ros2 topic hz /topic_name
```

Only call a service or send an action goal after confirming its name and type
from the running system. For example, `ros2 service call` takes a service name,
type, and YAML request; `ros2 action send_goal` takes an action name, type, and
YAML goal. The six-joint physical controller serves `/stop_robot` while running.

## ROS graph and multi-terminal setup

Every terminal that runs or inspects workspace packages needs the same
environment setup:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
```

ROS nodes on separate machines or containers must use compatible
`ROS_DOMAIN_ID` values and have network discovery configured. To isolate a
local experiment, choose one integer from 0 through 232 and export the same
value in every participating terminal, for example:

```bash
export ROS_DOMAIN_ID=23
```

Leave the default domain unless isolation is needed. This setting does not
replace network/firewall configuration.
