"""Static-block reaching task for demonstrating learned obstacle avoidance."""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

from gymnasium import spaces
import mujoco
import numpy as np

from reaching_env import ReachingEnv


@dataclass(frozen=True)
class CurriculumStage:
    min_distance: float
    max_distance: float
    min_start_clearance: float
    along: tuple
    lateral: tuple


# Meters; along is the fraction of the start-to-target XY segment.
# Stage 0 retains the original sampler; stage 1 is the active curriculum.
CURRICULUM = {
    1: CurriculumStage(0.10, 0.14, 0.04, (0.70, 0.95), (0.018, 0.040)),
}


class StaticObstacleEnv(ReachingEnv):
    """Reach randomized targets while avoiding a randomized static block."""

    obstacle_half_size = np.array([0.018, 0.018, 0.12])
    obstacle_center_z = 0.13
    min_start_clearance = 0.01  # meters between collision surfaces

    def _add_task_objects(self, xml):
        body = ET.SubElement(
            xml.find("worldbody"), "body", name="static_obstacle", mocap="true",
            pos="1.0 1.0 0.13",
        )
        ET.SubElement(
            body, "geom", name="static_obstacle_geom", type="box",
            size="0.018 0.018 0.12", rgba="0.9 0.25 0.12 1",
        )

    def __init__(
        self, render_mode=None, max_steps=200, *, curriculum_stage=0,
        clear_episode_prob=0.0,
    ):
        if curriculum_stage not in (0, *CURRICULUM):
            raise ValueError("Unknown obstacle curriculum stage.")
        if not 0 <= clear_episode_prob <= 1:
            raise ValueError("Clear episode probability must be in [0, 1].")
        self.curriculum_stage = curriculum_stage
        self.lesson = CURRICULUM.get(curriculum_stage)
        self.clear_episode_prob = clear_episode_prob
        self.obstacle_present = True
        kwargs = {}
        if self.lesson is not None:
            self.min_start_clearance = self.lesson.min_start_clearance
            kwargs = dict(min_target_distance=self.lesson.min_distance,
                          max_target_distance=self.lesson.max_distance)
        super().__init__(render_mode=render_mode, max_steps=max_steps, **kwargs)
        self.obstacle_mocap = int(self.model.body("static_obstacle").mocapid[0])
        self.block_geom = self.model.geom("static_obstacle_geom").id
        block = self.block_geom
        # The scene's other non-world, contact-compatible geoms belong to the arm.
        self.arm_geoms = np.flatnonzero(
            (self.model.geom_bodyid != 0)
            & (self.model.geom_bodyid != self.model.geom_bodyid[block])
            & (((self.model.geom_contype & self.model.geom_conaffinity[block])
                | (self.model.geom_conaffinity & self.model.geom_contype[block])) != 0)
        )
        self.obstacle_xy = np.array([0.25, -0.05], dtype=np.float64)
        low = np.array(
            [-1] * 5 + [0.12, -0.25, 0.0] + [0.036, 0.036, 0.24]
            + [0] * 2 + [0.12, -0.25, self.target_height] + [-1.0] * 3,
            dtype=np.float32,
        )
        high = np.array(
            [1] * 5 + [0.38, 0.15, 0.24] + [0.036, 0.036, 0.24]
            + [0] * 2 + [0.38, 0.15, self.target_height] + [1.0] * 3,
            dtype=np.float32,
        )
        if curriculum_stage or clear_episode_prob:
            low[5:11] = np.minimum(low[5:11], 0.0)
        self.observation_space = spaces.Box(low, high)

    def _observation(self):
        q = self.data.qpos[self.addresses[:5]]
        low, high = self.limits[:5].T
        joints = np.clip(2 * (q - low) / (high - low) - 1, -1, 1)

        if not self.obstacle_present:
            return super()._observation()

        # Obstacle center and full dimensions use the robot/world frame in meters.
        obstacle = np.array([
            self.obstacle_xy[0], self.obstacle_xy[1], self.obstacle_center_z,
            *(2 * self.obstacle_half_size),
        ], dtype=np.float64)
        return np.concatenate((
            joints, obstacle, np.zeros(2), self.goal, self.data.site_xpos[self.tip]
        )).astype(np.float32)

    def _reset_reaching(self, *, seed=None, options=None):
        # Parent reset parks the block via mj_resetData. Do not render until
        # the final obstacle layout is installed.
        mode = self.render_mode
        self.render_mode = None
        try:
            return super().reset(seed=seed, options=options)
        finally:
            self.render_mode = mode

    def _set_block(self, xy):
        self.obstacle_xy = np.asarray(xy, dtype=np.float64).copy()
        self.data.mocap_pos[self.obstacle_mocap] = [*xy, self.obstacle_center_z]
        mujoco.mj_forward(self.model, self.data)

    def _has_start_clearance(self):
        return not self._collision() and all(
            mujoco.mj_geomDistance(
                self.model, self.data, self.block_geom, int(geom),
                self.min_start_clearance, None,
            ) >= self.min_start_clearance for geom in self.arm_geoms
        )

    def _valid_layout(self, xy):
        self._set_block(xy)
        self._place(self.start_pose)
        if not self._has_start_clearance():
            return False
        self._place(self._goal_pose)
        return not self._collision()

    def _finish_reset(self, info):
        self._place(self.start_pose)
        self._previous_distance = self._distance()
        self._last_obs = self._observation()
        if self.render_mode == "human":
            self.render()
        return self._last_obs.copy(), info

    def place_planning_start(self, joint_positions, obstacle_xy, home_tolerance=0.0):
        """Verify and place the stationary measured arm and the training block."""
        self._done = True
        joints = np.asarray(joint_positions, dtype=np.float64)
        xy = np.asarray(obstacle_xy, dtype=np.float64)
        for name, value, size in (("joint_positions", joints, 6), ("obstacle_xy", xy, 2)):
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain {size} finite values.")
        if not np.isfinite(home_tolerance) or home_tolerance < 0:
            raise ValueError("home_tolerance must be finite and nonnegative.")
        if np.any(joints < self.limits[:, 0]) or np.any(joints > self.limits[:, 1]):
            raise ValueError("Measured joints are outside the model limits.")

        mujoco.mj_resetData(self.model, self.data)
        self._place(self.start_pose)
        if np.any(np.abs(joints - self.data.qpos[self.addresses]) > home_tolerance):
            raise ValueError("Measured joints are outside the training home tolerance.")
        self.data.qpos[self.addresses] = joints
        self.data.ctrl[self.actuators] = joints
        self.obstacle_present = True
        self._set_block(xy)
        if not self._has_start_clearance():
            raise ValueError("Starting scene is in contact or lacks obstacle clearance.")
        return self.data.site_xpos[self.tip].copy()

    def reset_for_planning(self, *, joint_positions, target_position, obstacle_xy,
                           home_tolerance=0.0):
        """Start a rollout from measured joints (radians) to a world target (meters)."""
        self.place_planning_start(joint_positions, obstacle_xy, home_tolerance)
        target = np.asarray(target_position, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_position must contain 3 finite values.")
        self.goal = target.copy()
        self.data.mocap_pos[self.mocap] = target
        mujoco.mj_forward(self.model, self.data)
        observation = self._observation()
        if not self.observation_space.contains(observation):
            raise ValueError("Planning scene is outside the policy observation bounds.")
        self._previous_distance = self._distance()
        if not self.min_target_distance <= self._previous_distance <= self.max_target_distance:
            raise ValueError("Target distance is outside the configured reaching range.")
        self._last_obs = observation
        self._steps, self._done = 0, False
        if self.render_mode == "human":
            self.render()
        return observation.copy(), self._info(False, False, False)

    def reset(self, *, seed=None, options=None):
        self.obstacle_present = True
        _, info = self._reset_reaching(seed=seed, options=options)
        if self.clear_episode_prob and self.np_random.random() < self.clear_episode_prob:
            self.obstacle_present = False
            self._set_block([1.0, 1.0])
            return self._finish_reset(info)
        if self.lesson is not None:
            # A tight target band or large start gap can make some targets
            # incompatible with a block. Resample the target, never relax the
            # requested clearance or silently substitute an easier episode.
            for attempt in range(64):
                if attempt:
                    _, info = self._reset_reaching(options=options)
                start = self.data.site_xpos[self.tip, :2].copy()
                delta = self.goal[:2] - start
                length = np.linalg.norm(delta)
                if length < 1e-8:
                    continue
                normal = np.array([-delta[1], delta[0]]) / length
                for _ in range(128):
                    along = self.np_random.uniform(*self.lesson.along)
                    lateral = self.np_random.uniform(*self.lesson.lateral)
                    lateral *= self.np_random.choice((-1, 1))
                    xy = start + along * delta + lateral * normal
                    if not (0.16 <= xy[0] <= 0.34 and -0.20 <= xy[1] <= 0.10):
                        continue
                    if self._valid_layout(xy):
                        return self._finish_reset(info)
            raise RuntimeError(
                f"No feasible layout for curriculum stage {self.curriculum_stage}."
            )
        return self._reset_baseline(info)

    def _reset_baseline(self, info):
        start_xy = self.data.site_xpos[self.tip, :2].copy()
        segment = self.goal[:2] - start_xy
        segment_length_sq = float(np.dot(segment, segment))

        # Prefer feasible layouts closest to the direct reach.
        best_xy = None
        best_clearance = float("inf")
        for _ in range(1000):
            self.obstacle_xy = self.np_random.uniform(
                np.array([0.16, -0.20]), np.array([0.34, 0.10])
            )
            along = np.clip(
                np.dot(self.obstacle_xy - start_xy, segment) / segment_length_sq,
                0.0, 1.0,
            )
            nearest = start_xy + along * segment
            clearance = float(np.linalg.norm(self.obstacle_xy - nearest))
            if clearance >= best_clearance:
                continue
            if self._valid_layout(self.obstacle_xy):
                best_xy = self.obstacle_xy.copy()
                best_clearance = clearance

        if best_xy is None:
            raise RuntimeError("Could not sample a collision-free obstacle layout.")
        self._set_block(best_xy)
        return self._finish_reset(info)


class JawObstacleEnv(StaticObstacleEnv):
    """Report the jaw-to-block surface vector in world coordinates (meters)."""

    def __init__(self, render_mode=None, max_steps=200, **kwargs):
        super().__init__(render_mode=render_mode, max_steps=max_steps, **kwargs)
        jaw = self.model.body("moving_jaw_so101_v1").id
        self.jaw_geom, = self.arm_geoms[self.model.geom_bodyid[self.arm_geoms] == jaw]
        low = self.observation_space.low.copy()
        high = self.observation_space.high.copy()
        low[8:11], high[8:11] = -2.0, 2.0
        self.observation_space = spaces.Box(low, high)

    def _observation(self):
        obs = super()._observation()
        if not self.obstacle_present:
            return obs
        points = np.zeros(6)
        gap = mujoco.mj_geomDistance(
            self.model, self.data, self.jaw_geom, self.block_geom, 2.0, points
        )
        obs[8:11] = points[3:] - points[:3] if gap > 0 else 0.0
        return obs


TASK_ENVS = {
    "reaching": ReachingEnv,
    "static-obstacle": StaticObstacleEnv,
    "jaw-obstacle": JawObstacleEnv,
}
