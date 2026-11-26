"""Stage A: simulated reaching with joint state and ideal overhead bounding boxes."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


class ReachingEnv(gym.Env):
    """No hardware access. The imported dynamics and camera pose are provisional."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None, max_steps=200):
        # Validate the render mode and episode length.
        if render_mode not in (None, "human") or max_steps < 1:
            raise ValueError("Use render_mode=None/'human' and max_steps >= 1.")
        self.render_mode, self.max_steps = render_mode, max_steps

        # Load the robot scene before adding the target marker.
        root = Path(__file__).resolve().parents[1] / "models/SO101"
        xml = ET.fromstring((root / "scene.xml").read_text())
        
        # The target is a visual marker, so reaching it never causes a collision.
        marker = ET.SubElement(xml.find("worldbody"), "body",
                               name="reaching_target", mocap="true")
        self.half_size = np.array([0.01, 0.01, 0.002])
        ET.SubElement(marker, "geom", type="box", size="0.01 0.01 0.002",
                      rgba="0 1 0 1", contype="0", conaffinity="0")
        
        # Bundle model files and meshes for MuJoCo.
        assets = {
            p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*") if p.suffix.lower() in (".xml", ".stl")
        }
        self.model = mujoco.MjModel.from_xml_string(
            ET.tostring(xml, encoding="unicode"), assets
        )
        self.data = mujoco.MjData(self.model)
        
        # Look up joint positions and their matching actuators.
        names = (*ARM, "gripper")
        joints = [self.model.joint(n).id for n in names]
        self.addresses = self.model.jnt_qposadr[joints]
        self.actuators = [self.model.actuator(n).id for n in names]
        
        # Model angles and commands are radians, independent of hardware units.
        self.limits = np.array([
            (max(self.model.jnt_range[j, 0], self.model.actuator_ctrlrange[a, 0]),
             min(self.model.jnt_range[j, 1], self.model.actuator_ctrlrange[a, 1]))
            for j, a in zip(joints, self.actuators)
        ])
        
        # Cache scene IDs and set the control interval.
        self.tip = self.model.site("gripperframe").id
        self.camera = self.model.camera("overhead").id
        self.mocap = int(self.model.body("reaching_target").mocapid[0])
        self.substeps = round(0.05 / self.model.opt.timestep)
        self.control_dt = self.substeps * self.model.opt.timestep
        
        # Initial target plane: 15 cm above the model floor; calibrate later.
        self.target_height, self.tolerance = 0.15, 0.02
        
        # World-coordinate rectangle represented by the ideal image, in meters.
        # Approximately matches the current straight-down view at target height.
        # Calibrate these bounds against the real overhead camera before transfer.
        self.image_bounds = np.array([[-0.50, 0.50], [-0.525, 0.225]])
        self.action_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
        
        # Layout: joints[5], obstacle box[4], obstacle delta[4], target box[4].
        low = np.array([-1] * 5 + [0] * 4 + [-1] * 4 + [0] * 4, np.float32)
        self.observation_space = spaces.Box(low, np.ones(17, np.float32))
        self._viewer, self._done = None, True

    def _place(self, angles):
        # Set the arm pose, hold the gripper closed, and clear motion.
        self.data.qpos[self.addresses[:5]] = angles
        self.data.qpos[self.addresses[5]] = 0
        self.data.qvel[:] = 0
        self.data.ctrl[self.actuators] = self.data.qpos[self.addresses]
        mujoco.mj_forward(self.model, self.data)

    def _collision(self):
        # Includes floor, self, wrist-camera and mount contacts. The imported
        # base has no collision geometry; native parent/child filters apply.
        return any(c.dist <= 0 for c in self.data.contact)

    def _box(self, position):
        # Directly map the target's XY position into [cx, cy, width, height].
        # This assumes a fixed target height and a straight-down view:
        # +X points right, +Y points up, and image y grows downward.
        # No image rendering or 3D corner projection is needed.
        (xmin, xmax), (ymin, ymax) = self.image_bounds
        width, height = xmax - xmin, ymax - ymin
        x, y = position[:2]
        box = np.array([
            (x - xmin) / width,
            (ymax - y) / height,
            2 * self.half_size[0] / width,
            2 * self.half_size[1] / height,
        ], dtype=np.float32)
        # Keep only targets whose full box lies inside the normalized image.
        low, high = box[:2] - box[2:] / 2, box[:2] + box[2:] / 2
        return box if np.all(low >= 0) and np.all(high <= 1) else None

    def _observation(self):
        # Normalize arm joint positions to [-1, 1].
        q = self.data.qpos[self.addresses[:5]]
        low, high = self.limits[:5].T
        joints = np.clip(2 * (q - low) / (high - low) - 1, -1, 1)
        # Zero boxes reserve the future obstacle fields without inventing one.
        return np.concatenate((joints, np.zeros(8), self.target_box)).astype(np.float32)

    def _distance(self):
        # Measure the end effector's 3D distance to the target.
        return float(np.linalg.norm(self.data.site_xpos[self.tip] - self.goal))

    def reset(self, *, seed=None, options=None):
        # Seed sampling and clear the previous simulation state.
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        low, high = self.limits[:5].T
        # Rejection sampling uses simulator geometry only for task generation.
        # A nearby collision-free pose witnesses reachability within tolerance;
        # it does not guarantee a collision-free path from every random start.
        for _ in range(10000):
            angles = self.np_random.uniform(low + 0.05, high - 0.05)
            self._place(angles)
            tip = self.data.site_xpos[self.tip].copy()
            if self._collision() or abs(tip[2] - self.target_height) > 0.005:
                continue
            if not (0.12 < tip[0] < 0.38 and -0.25 < tip[1] < 0.15):
                continue
            tip[2] = self.target_height
            box = self._box(tip)
            if box is not None:
                self.goal, self.target_box = tip, box
                self._goal_pose = angles.copy()
                break
        else:
            raise RuntimeError("Could not sample a reachable, visible target.")
        self.data.mocap_pos[self.mocap] = self.goal
        # Sample a safe starting pose away from the target.
        for _ in range(10000):
            self._place(self.np_random.uniform(low + 0.05, high - 0.05))
            if not self._collision() and self._distance() > 3 * self.tolerance:
                break
        else:
            raise RuntimeError("Could not sample a collision-free starting pose.")
        # Initialize episode tracking and the first observation.
        self._steps, self._done = 0, False
        self._previous_distance = self._distance()
        self._last_obs = self._observation()
        if self.render_mode == "human":
            self.render()
        return self._last_obs.copy(), self._info(False, False, False)

    def _status(self):
        # Check simulation values and joint limits.
        q = self.data.qpos[self.addresses]
        finite = all(np.isfinite(x).all() for x in (
            self.data.qpos, self.data.qvel, self.data.qacc
        ))
        limits = np.any(q < self.limits[:, 0] - 0.001) or np.any(
            q > self.limits[:, 1] + 0.001
        )
        # This is an actual-speed guard, separate from the command-rate limit.
        unsafe = not finite or limits or np.any(np.abs(self.data.qvel) > 1.5)
        return bool(self._collision()), bool(unsafe)

    def _info(self, success, collision, unsafe):
        # Reward-only geometry stays in diagnostics, never in policy observations.
        return {"is_success": bool(success), "collision": bool(collision),
                "unsafe": bool(unsafe), "distance": float(self._previous_distance)}

    def step(self, action):
        # Require an active episode and five finite action values.
        if self._done:
            raise gym.error.ResetNeeded("Call reset() before the next step.")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("Action must contain five finite values.")
        with (self._viewer.lock() if self._viewer else nullcontext()):
            q = self.data.qpos[self.addresses[:5]].copy()
            # Measured-position increments: <=0.02 rad/step and <=0.5 rad/s.
            increment = min(0.02, 0.5 * self.control_dt)
            target = np.clip(
                q + increment * np.clip(action, -1, 1), *self.limits[:5].T
            )
            self.data.ctrl[self.actuators[5]] = 0
            collision, unsafe = self._status()
            for substep in range(self.substeps):
                if collision or unsafe:
                    break
                # Ramp the command across physics steps rather than applying
                # a sudden position jump to the stiff simulated servo.
                fraction = (substep + 1) / self.substeps
                self.data.ctrl[self.actuators[:5]] = q + fraction * (target - q)
                before = self.data.time
                mujoco.mj_step(self.model, self.data)
                # Refresh contact geometry after integration, checking every
                # physics step so a brief contact is not missed between actions.
                mujoco.mj_forward(self.model, self.data)
                collision, unsafe = self._status()
                unsafe = unsafe or self.data.time <= before
            # Preserve the last distance if simulation values become invalid.
            finite = np.isfinite(self.data.qpos).all() and np.isfinite(
                self.data.site_xpos[self.tip]
            ).all()
            distance = self._distance() if finite else self._previous_distance
            success = not (collision or unsafe) and distance <= self.tolerance
            # Initial reward weights: progress in meters, time cost, and
            # terminal bonuses/penalties. These remain tuning parameters.
            reward = 10 * (self._previous_distance - distance) - 0.01
            reward += 10 * success - 10 * bool(collision or unsafe)
            self._previous_distance = distance
            # Track episode completion and the time limit.
            self._steps += 1
            terminated = bool(success or collision or unsafe)
            # A time limit is truncation, not a failed terminal outcome.
            truncated = bool(self._steps >= self.max_steps and not terminated)
            self._done = terminated or truncated
            if finite:
                self._last_obs = self._observation()
        if self.render_mode == "human":
            self.render()
        return (self._last_obs.copy(), float(reward), terminated, truncated,
                self._info(success, collision, unsafe))

    def render(self):
        # Open the viewer on demand using the overhead camera.
        if self.render_mode != "human":
            return
        if self._viewer is None:
            from mujoco import viewer
            self._viewer = viewer.launch_passive(self.model, self.data)
            with self._viewer.lock():
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self._viewer.cam.fixedcamid = self.camera
        if self._viewer.is_running():
            self._viewer.sync()

    def close(self):
        # Release the viewer when the environment closes.
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def main():
    # Choose whether to show the simulation viewer.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    env = ReachingEnv(render_mode=None if args.headless else "human")
    try:
        obs, info = env.reset(seed=0)
        env.action_space.seed(0)
        print("Observation:", obs.shape, "Action:", env.action_space.shape)
        print("Random-action demonstration; this is not a trained policy.")
        # Run random actions and restart completed episodes.
        for _ in range(200):
            started = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                print("Episode ended:", info, "truncated:", truncated)
                env.reset()
            if not args.headless:
                # Match the demo playback to the control interval.
                if not env._viewer.is_running():
                    break
                time.sleep(max(0, env.control_dt - (time.perf_counter() - started)))
        print("Demonstration complete.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
