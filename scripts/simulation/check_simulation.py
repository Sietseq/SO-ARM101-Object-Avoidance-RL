"""Inspect SO101 joint motion in simulation; no robot connection is used."""

import argparse
from contextlib import nullcontext
import math
import time

import mujoco
import numpy as np

from scripts import PROJECT_ROOT

ARM_JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--free-camera", action="store_true")
    args = parser.parse_args()
    scene = PROJECT_ROOT / "models/SO101/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    names = (*ARM_JOINTS, "gripper")
    joint_ids = [model.joint(name).id for name in names]
    actuator_ids = [model.actuator(name).id for name in names]
    addresses = model.jnt_qposadr[joint_ids]
    limits = np.array([
        (max(model.jnt_range[j, 0], model.actuator_ctrlrange[a, 0]),
         min(model.jnt_range[j, 1], model.actuator_ctrlrange[a, 1]))
        for j, a in zip(joint_ids, actuator_ids)
    ])
    home = np.clip(model.qpos0[addresses], limits[:, 0], limits[:, 1])
    data.qpos[addresses] = home
    data.ctrl[actuator_ids] = home
    mujoco.mj_forward(model, data)
    tip = model.site("gripperframe").id
    print("Simulation only. Angles are degrees; site positions are meters.")
    for name, (low, high) in zip(names, limits):
        print(f"{name:16s} limits: {math.degrees(low):7.2f} to {math.degrees(high):7.2f}")
    print("gripperframe is the imported end-effector reference; verify its physical placement.")
    print("Each arm joint sweeps +/-10 degrees over 4 seconds; gripper target stays fixed.")
    session = nullcontext(None)
    if not args.headless:
        from mujoco import viewer as mj_viewer
        session = mj_viewer.launch_passive(model, data)
    max_error = np.zeros(len(names))
    previous = -1
    with session as viewer:
        if viewer:
            with viewer.lock():
                viewer.opt.sitegroup[3] = 1
                if not args.free_camera:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = model.camera("overhead").id
        steps = math.ceil(20.0 / model.opt.timestep)
        started = time.perf_counter()
        for step in range(steps):
            if viewer and not viewer.is_running():
                break
            active = min(int(data.time / 4.0), len(ARM_JOINTS) - 1)
            if active != previous:
                print(f"Moving {ARM_JOINTS[active]}")
                previous = active
            with (viewer.lock() if viewer else nullcontext()):
                target = home.copy()
                phase = (data.time % 4.0) / 4.0
                target[active] += math.radians(10) * math.sin(2 * math.pi * phase) ** 3
                data.ctrl[actuator_ids] = np.clip(target, limits[:, 0], limits[:, 1])
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError("Simulation produced a non-finite joint state.")
                max_error = np.maximum(
                    max_error, np.abs(data.ctrl[actuator_ids] - data.qpos[addresses])
                )
            if viewer:
                if step % max(1, round(1 / (60 * model.opt.timestep))) == 0:
                    viewer.sync()
                deadline = started + (step + 1) * model.opt.timestep
                time.sleep(max(0, deadline - time.perf_counter()))
    mujoco.mj_forward(model, data)
    print(f"Simulated {data.time:.2f} seconds.")
    for name, error in zip(names, max_error):
        print(f"{name:16s} max tracking error: {math.degrees(error):.3f} degrees")
    print("Final gripperframe position:", np.round(data.site_xpos[tip], 4))


if __name__ == "__main__":
    main()
