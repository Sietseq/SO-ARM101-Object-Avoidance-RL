# Project Specification

## Sim-to-Real Reinforcement Learning for Vision-Based Dynamic Obstacle Avoidance

**Version:** 1.0  
**Project type:** Robotics / Reinforcement Learning / Sim-to-Real  
**Hardware:** SO-ARM101 robotic arm  
**Simulation:** MuJoCo  
**RL algorithm:** Proximal Policy Optimization (PPO)  
**Estimated scope:** 1–2 weeks for a proof of concept, subject to training and integration results

---

# 1. Project Overview

## 1.1 Objective

Develop a reinforcement learning policy that enables an SO-ARM101 robotic arm to navigate its end effector toward a randomly positioned target while avoiding a moving obstacle.

The policy will be trained in simulation and subsequently deployed on the physical robot without retraining as the initial Sim-to-Real experiment.

The system will use:

- Robot joint positions (proprioception)
- Target bounding-box coordinates
- Obstacle bounding-box coordinates
- Temporal changes in obstacle detections

The policy will output joint-space movement commands.

**The central hypothesis is that an RL policy can learn effective whole-arm obstacle avoidance using joint positions and 2D visual observations without explicitly receiving the robot's kinematic or collision geometry.**

## 1.2 Research Questions

**RQ1:** Can a neural-network policy learn to navigate the robot to randomly positioned targets using joint positions and visual observations?

**RQ2:** Can the policy learn whole-arm collision avoidance without explicit robot link geometry as an observation?

**RQ3:** Does temporal obstacle information improve avoidance of moving obstacles?

**RQ4:** How effectively does a policy trained with simulated bounding boxes transfer to a physical robot using real camera detections?

---

# 2. Project Scope

## 2.1 In Scope

- SO-ARM101 simulation in MuJoCo
- Gymnasium-compatible RL environment
- PPO training using Stable-Baselines3
- Randomized robot starting configurations
- Randomized target positions
- One moving obstacle
- Infinite-height obstacle collision approximation
- Joint-space policy control
- Fixed overhead camera
- Bounding-box-based visual observations
- Basic domain randomization
- Real robot policy deployment
- Simulation and physical evaluations

## 2.2 Out of Scope

- Grasping and object manipulation
- Multiple simultaneous obstacles
- End-to-end raw-image reinforcement learning
- Custom motorized obstacle mechanisms
- Explicit inverse-kinematics policy control
- Training a custom object detection neural network
- Learning a full physics or geometric model of the robot
- Real-world online reinforcement learning

## 2.3 Key Assumptions

1. The robot is mounted in a fixed position.
2. The overhead camera remains fixed throughout deployment.
3. The target lies on a known, fixed-height plane.
4. Only one obstacle is present.
5. The obstacle's physical footprint is approximately rectangular.
6. The gripper remains in a fixed configuration.
7. The target and obstacle are distinguishable through visual detection.
8. The physical test area is controlled and supervised.

---

# 3. System Architecture

## 3.1 Simulation

**Components:**

- MuJoCo physics engine
- SO-ARM101 robot model
- Fixed overhead virtual camera
- Randomly positioned target
- Moving rectangular obstacle
- Collision detection
- Gymnasium environment
- PPO agent

**Simulation loop:**

1. Reset the environment.
2. Randomize valid robot joint positions and target location.
3. Initialize the obstacle position and motion.
4. Generate the simulated visual observations.
5. Construct the RL observation vector.
6. Query the policy for an action.
7. Apply joint-position commands.
8. Advance the simulation.
9. Calculate reward and termination conditions.
10. Repeat until the episode ends.

## 3.2 Real-World Deployment

**Components:**

- SO-ARM101 robotic arm
- Fixed overhead RGB camera
- Lightweight, colored foam obstacle
- Visually identifiable target
- OpenCV-based object detection
- Joint-state feedback
- Trained PPO policy
- Joint controller and independent safety mechanisms

**Deployment loop:**

1. Capture the overhead camera frame.
2. Detect the target and obstacle.
3. Extract normalized bounding boxes.
4. Estimate obstacle movement from consecutive observations.
5. Read robot joint positions.
6. Construct the observation vector.
7. Execute policy inference.
8. Validate and limit commanded movement.
9. Send joint commands to the robot.
10. Repeat at a controlled frequency.

---

# 4. Environment Specification

## 4.1 Robot

**Model:** SO-ARM101

The robot's five arm joints are controlled by the policy. The gripper is held at a fixed configuration.

The MuJoCo model must include suitable collision geometry for all relevant robot links.

## 4.2 Target

The target is represented visually in the overhead camera.

**Properties:**

- Randomized location within a predefined reachable region
- Fixed height
- Visually distinct from the obstacle
- No grasping required

An episode succeeds when the end effector reaches a specified distance tolerance from the target's 3D reference position.

The end-effector position is available to the simulator for reward calculation, but is not directly supplied to the policy.

## 4.3 Obstacle

The obstacle is a rectangular object moving within the robot's workspace.

**Collision representation:**

The obstacle is approximated by a vertical collision column with a constant 2D footprint extending through the robot's operational height range.

This is equivalent to treating the obstacle as having infinite vertical extent for the task's collision calculations.

The visible object itself retains realistic dimensions.

**Movement characteristics:**

- Linear horizontal motion initially
- Randomized initial position
- Randomized speed
- Randomized direction
- Optional pauses and reversals

The obstacle must be positioned so valid, collision-free reaching solutions exist.

During physical deployment, the obstacle will be moved manually using a long, lightweight rod, keeping the operator's hands outside the robot workspace.

## 4.4 Collision Detection

Collision checks will consider the entire robot arm, not only the end effector.

A collision is registered if any relevant robot link intersects the obstacle's vertical exclusion column.

The environment must also handle joint-limit violations, self-collisions, and unsafe table contact.

For real-world deployment, an independent safety monitor will use robot geometry and the obstacle's estimated tabletop footprint to enforce conservative collision boundaries.

**Important distinction:** The RL policy does not receive link geometry, but the simulator and safety monitor are allowed to use it.

---

# 5. Reinforcement Learning Specification

## 5.1 Learning Algorithm

**Algorithm:** Proximal Policy Optimization (PPO)

**Library:** Stable-Baselines3

**Policy architecture:** Multilayer perceptron (MLP)

PPO is selected because it provides a well-established continuous-action RL baseline and is straightforward to integrate with Gymnasium environments.

## 5.2 Observation Space

The initial observation vector contains 17 values:

$$
o_t=[q_{1:5},b_o,\Delta b_o,b_g]
$$

| Component             | Dimensions |
| --------------------- | ---------: |
| Arm joint positions   |          5 |
| Obstacle bounding box |          4 |
| Bounding-box changes  |          4 |
| Target bounding box   |          4 |
| **Total**             |     **17** |

Bounding boxes are represented as:

$$
b=[c_x,c_y,w,h]
$$

All image coordinates and dimensions are normalized by image width and height.

Joint positions are normalized using their valid ranges. Temporal differences are calculated using a consistent observation interval.

The initial observation design assumes that slow control and fixed target height make the task sufficiently observable. Joint velocities or additional observation history may be introduced if necessary.

No explicit forward kinematics, robot-link positions, link-to-obstacle distances, or planned trajectories will be provided to the policy.

## 5.3 Action Space

The policy outputs continuous joint-position increments:

$$
a_t\in[-1,1]^5
$$

Joint commands are computed as:

$$
q_{\mathrm{cmd}}=\operatorname{clip}(q_t+\alpha a_t,q_{\min},q_{\max})
$$

Where:

- $q_t$: current measured joint positions
- $\alpha$: maximum commanded increment
- $q_{\min},q_{\max}$: joint position limits

Commands must also satisfy velocity and safety constraints.

## 5.4 Reward Function

The initial reward function is:

$$
r_t=
w_p(d_{t-1}-d_t)
+w_sI_{\text{success}}
-w_cI_{\text{collision}}
-w_t
$$

Where:

- $d_t$: end-effector distance to the target
- $I_{\text{success}}$: target reached
- $I_{\text{collision}}$: collision occurred
- $w_p$: progress reward coefficient
- $w_s$: success bonus
- $w_c$: collision penalty
- $w_t$: per-step time penalty

Reward coefficients will be tuned experimentally.

The reward should encourage successful, safe, efficient reaching without forcing the robot to move continuously. Waiting for an obstacle to pass is an acceptable learned behavior.

## 5.5 Episode Termination

Episodes terminate when:

1. The target is successfully reached.
2. A collision occurs.
3. An unsafe state is detected.
4. The maximum episode duration is reached.

Success and collision are terminal outcomes. Reaching the time limit is treated as truncation when appropriate.

---

# 6. Training Strategy

Training will follow a curriculum of progressively harder tasks.

## Stage A — Reaching

**Environment:** No obstacle.

**Objective:** Learn joint-space control for target reaching.

**Randomization:**

- Initial joint configuration
- Target position

**Completion criterion:** The policy demonstrates reliable reaching on unseen valid configurations.

## Stage B — Static Obstacle

**Environment:** One stationary obstacle.

**Objective:** Learn whole-arm collision avoidance.

**Randomization:**

- Initial joint configuration
- Target position
- Obstacle position

**Completion criterion:** The policy improves collision-free reaching performance over a reaching-only baseline.

## Stage C — Dynamic Obstacle

**Environment:** One moving obstacle.

**Objective:** Learn reactive avoidance behavior.

**Randomization:**

- Obstacle position
- Movement speed
- Movement direction
- Movement start time
- Optional pauses and reversals

**Completion criterion:** The policy achieves meaningful success rates on previously unseen moving-obstacle scenarios.

## Stage D — Domain Randomization

Introduce controlled variations in:

- Camera alignment and projection parameters
- Bounding-box position and size noise
- Missed detections
- Observation latency
- Joint measurement noise
- Actuator response and control delay
- Obstacle movement profiles

The aim is to reduce the difference between simulated observations and real-world observations without making early training unnecessarily difficult.

---

# 7. Vision System

## 7.1 Camera

A fixed overhead RGB camera observes the tabletop workspace.

The virtual camera will be configured to approximate the real camera's:

- Mounting height
- Orientation
- Field of view
- Intrinsic parameters
- Image resolution or normalized coordinate conventions

## 7.2 Object Detection

Initial implementation uses OpenCV color segmentation.

The target and obstacle will have distinct colors.

The detector outputs:

$$
[c_x,c_y,w,h]
$$

More sophisticated detection algorithms may be introduced later, but are not required for the first demonstration.

## 7.3 Simulated Bounding Boxes

Simulated bounding boxes must be produced from the same type of image-space projection as the real camera detections.

The simulator may use known object geometry to produce perfect projected boxes during early training.

Subsequent experiments will introduce realistic detection errors.

## 7.4 Temporal Observations

Obstacle motion is estimated using consecutive detections:

$$
\Delta b_t=b_t-b_{t-1}
$$

The difference is associated with a consistent observation interval.

The system must define behavior when detections are temporarily unavailable, such as holding the last observation with a detection-validity mechanism and stopping the physical arm when reliable tracking is lost.

---

# 8. Sim-to-Real Transfer

## 8.1 Transfer Strategy

Train in simulation and deploy the same trained policy on the physical robot.

The observation and action interfaces must remain consistent.

## 8.2 Calibration

Before deployment:

- Verify joint ordering.
- Verify joint units, directions, and zero offsets.
- Check the simulated and real joint limits.
- Match the overhead camera projection.
- Verify target and obstacle detection.
- Match control timing.
- Verify physical stopping behavior.

## 8.3 Safety

Real-world experiments must use:

- Conservative joint-position and velocity limits
- Independent collision monitoring
- Accessible emergency stop
- Lightweight obstacle materials
- Reduced initial control speeds
- Supervised operation
- A clearly defined, restricted workspace
- Stop-on-detection-failure behavior

The first deployment tests must occur without moving obstacles.

The physical robot must not rely on the RL policy alone for safe operation.
