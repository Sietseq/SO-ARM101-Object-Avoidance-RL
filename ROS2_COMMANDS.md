# ROS 2 command reference

This guide records the ROS 2 commands for this repository's `ros2_ws` workspace.
The checked-in workspace is a scaffold: `so101_demo` currently installs only a
placeholder `planner_node`, `so101_interfaces` defines no custom interfaces,
and there is no launch file yet. Commands that assume the camera, robot, planner
services, actions, or demo launch file exist are not usable until those are
implemented.

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

## Run the current scaffold

The only registered executable currently available is the planner placeholder:

```bash
ros2 run so101_demo planner_node
```

It prints a greeting and exits; it is not yet a ROS node. Confirm registered
executables with:

```bash
ros2 pkg executables so101_demo
```

There is no `demo.launch.py` in the current package, so the launch command in
`PROJECT_SPECS.md` is a proposed future interface, not a working command yet.

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
YAML goal. The `/start_demo`, `/execute_plan`, `/return_home`, `/stop_robot`,
and `/robot/move_to_home` interfaces mentioned in `PROJECT_SPECS.md` are
planned interfaces and are not implemented in this scaffold.

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
