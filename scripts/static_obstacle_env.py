"""Stage B: fixed-start reaching around a stationary vertical exclusion column."""

import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np

from reaching_env import ReachingEnv


class StaticObstacleEnv(ReachingEnv):
    """Same controls and observations as Stage A, with whole-arm avoidance."""

    success_bonus = 50.0

    # Meters. The visible box is 2 x 2 x 4 cm; height never permits a bypass.
    box_xy = np.array([0.28, -0.06])
    box_half = np.array([0.01, 0.01])

    def _add_task_objects(self, xml):
        body = ET.SubElement(
            xml.find("worldbody"), "body", name="static_obstacle",
            mocap="true", pos="2 2 0.5",
        )
        # Covers all heights the joint-limited arm can reach.
        ET.SubElement(
            body, "geom", name="obstacle_column", type="box",
            size="0.01 0.01 1", rgba="0 0 0 0", group="5",
        )
        ET.SubElement(
            body, "geom", name="obstacle_visible", type="box",
            pos="0 0 -0.48", size="0.01 0.01 0.02",
            rgba="1 0.25 0.05 1", contype="0", conaffinity="0",
        )

    def __init__(self, render_mode=None, max_steps=200, *, pool_size=512,
                 pool_seed=0, blocked_fraction=1.0):
        if not isinstance(pool_size, int) or pool_size < 0:
            raise ValueError("pool_size must be a nonnegative integer.")
        if not isinstance(pool_seed, int) or pool_seed < 0:
            raise ValueError("pool_seed must be a nonnegative integer.")
        if not 0.0 <= blocked_fraction <= 1.0:
            raise ValueError("blocked_fraction must be between zero and one.")
        self.blocked_fraction = blocked_fraction
        self.pool_size, self.pool_seed = pool_size, pool_seed
        self._layout_pool = None
        self._box_observation = np.zeros(4, dtype=np.float32)
        super().__init__(render_mode=render_mode, max_steps=max_steps)
        self.box_mocap = int(self.model.body("static_obstacle").mocapid[0])

    def _observation(self):
        obs = super()._observation()
        obs[5:9] = self._box_observation
        # A stationary obstacle has zero bounding-box changes.
        return obs

    def _project_box(self):
        # Project all eight visible-box corners using the actual scene camera.
        corners = np.array([
            [x, y, z]
            for x in self.box_xy[0] + np.array([-0.01, 0.01])
            for y in self.box_xy[1] + np.array([-0.01, 0.01])
            for z in (0.0, 0.04)
        ])
        rotation = self.data.cam_xmat[self.camera].reshape(3, 3)
        local = (corners - self.data.cam_xpos[self.camera]) @ rotation
        depth = -local[:, 2]
        if np.any(depth <= 0):
            raise RuntimeError("Obstacle is behind the overhead camera.")
        # Square image; camera fovy defines both axes for aspect ratio one.
        scale = 2 * depth * np.tan(np.deg2rad(self.model.cam_fovy[self.camera]) / 2)
        uv = np.column_stack((0.5 + local[:, 0] / scale,
                              0.5 - local[:, 1] / scale))
        if np.any(uv < 0) or np.any(uv > 1):
            raise RuntimeError("Obstacle is outside the overhead camera view.")
        low, high = uv.min(axis=0), uv.max(axis=0)
        return np.concatenate(((low + high) / 2, high - low)).astype(np.float32)

    def _blocks_segment(self, start, goal):
        # Slab intersection of the straight XY route with the box footprint.
        delta = goal[:2] - start[:2]
        enter, leave = 0.0, 1.0
        for axis in range(2):
            low, high = self.box_xy[axis] + np.array([-1, 1]) * self.box_half[axis]
            if abs(delta[axis]) < 1e-12:
                if not low <= start[axis] <= high:
                    return False
                continue
            a, b = sorted(((low - start[axis]) / delta[axis],
                           (high - start[axis]) / delta[axis]))
            enter, leave = max(enter, a), min(leave, b)
        return 0 < enter <= leave < 1

    def _clear_segment(self, start, end):
        # Geometry-only task validation; these poses never enter observations.
        count = max(1, int(np.ceil(np.max(np.abs(end - start)) / 0.04)))
        for pose in np.linspace(start, end, count + 1):
            self._place(pose)
            if self._collision():
                return False
        return True

    def _build_layout_pool(self):
        blocked_count = round(self.pool_size * self.blocked_fraction)
        print(f"Precomputing {self.pool_size} layouts "
              f"({blocked_count} blocked, seed {self.pool_seed})...", flush=True)
        sampler = StaticObstacleEnv(max_steps=self.max_steps, pool_size=0)
        layouts = []
        try:
            for index in range(self.pool_size):
                sampler.blocked_fraction = 1.0 if index < blocked_count else 0.0
                sampler.reset(seed=self.pool_seed if index == 0 else None)
                layouts.append((
                    sampler.data.qpos[sampler.addresses[:5]].copy(),
                    sampler.goal.copy(), sampler._goal_pose.copy(),
                    sampler._waypoint_pose.copy(),
                ))
                if (index + 1) % 128 == 0 or index + 1 == self.pool_size:
                    print(f"Validated {index + 1}/{self.pool_size} layouts.",
                          flush=True)
        finally:
            sampler.close()
        self._layout_pool = tuple(layouts)

    def reset(self, *, seed=None, options=None):
        if self.pool_size == 0:
            return self._sample_layout(seed=seed, options=options)
        gym.Env.reset(self, seed=seed)
        if self._layout_pool is None:
            self._build_layout_pool()
        index = int(self.np_random.integers(len(self._layout_pool)))
        start, goal, goal_pose, waypoint = self._layout_pool[index]
        mujoco.mj_resetData(self.model, self.data)
        self.goal = goal.copy()
        self._goal_pose = goal_pose.copy()
        self._waypoint_pose = waypoint.copy()
        self.data.mocap_pos[self.mocap] = self.goal
        self.data.mocap_pos[self.box_mocap] = [*self.box_xy, 0.5]
        self._place(start)
        self._box_observation = self._project_box()
        self._previous_distance = self._distance()
        self._steps, self._done = 0, False
        self._last_obs = self._observation()
        if self.render_mode == "human":
            self.render()
        return self._last_obs.copy(), self._info(False, False, False)

    def _sample_layout(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        require_blocked = self.np_random.random() < self.blocked_fraction
        # Stage A samples targets with the column parked out of the workspace.
        render_mode = self.render_mode
        self.render_mode = None
        self._box_observation[:] = 0
        try:
            for attempt in range(200):
                super().reset(options=options)
                start = self.data.qpos[self.addresses[:5]].copy()
                start_tip = self.data.site_xpos[self.tip].copy()
                goal_pose = self._goal_pose.copy()
                if self._blocks_segment(start_tip, self.goal) != require_blocked:
                    continue
                # Keep the entire success ball clear of the exclusion footprint.
                clearance = np.maximum(np.abs(self.goal[:2] - self.box_xy)
                                       - self.box_half, 0)
                if np.linalg.norm(clearance) <= self.tolerance:
                    continue
                self.data.mocap_pos[self.box_mocap] = [*self.box_xy, 0.5]
                self._place(start)
                if self._collision():
                    raise RuntimeError("Fixed obstacle intersects the fixed start.")
                self._place(goal_pose)
                if self._collision():
                    continue
                # Clear-route targets use a direct joint interpolation;
                # blocked targets require a sampled detour.
                low, high = self.limits[:5].T
                middle = (start + goal_pose) / 2
                for _ in range(32 if require_blocked else 1):
                    waypoint = (
                        self.np_random.uniform(
                            np.maximum(low + 0.05, middle - 0.8),
                            np.minimum(high - 0.05, middle + 0.8),
                        ) if require_blocked else middle
                    )
                    if (self._clear_segment(start, waypoint)
                            and self._clear_segment(waypoint, goal_pose)):
                        self._waypoint_pose = waypoint.copy()
                        break
                else:
                    continue
                self._place(start)
                self._box_observation = self._project_box()
                self._previous_distance = self._distance()
                self._steps, self._done = 0, False
                self._last_obs = self._observation()
                break
            else:
                self._done = True
                raise RuntimeError("Could not sample a target with a clear detour.")
        finally:
            self.render_mode = render_mode
        if self.render_mode == "human":
            self.render()
        return self._last_obs.copy(), self._info(False, False, False)
