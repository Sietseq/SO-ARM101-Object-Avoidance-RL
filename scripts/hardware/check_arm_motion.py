
import os
from pathlib import Path
import time

from lerobot.robots.so_follower import (
    SO101Follower,
    SO101FollowerConfig,
)

from scripts.so101.relative_move import TargetNotReached, get_positions, move_relative as move_steps

# ==============================
# Robot configuration
# ==============================

calibration_path = Path(
    os.environ["FOLLOWER_CALIBRATION"]
).expanduser()

if calibration_path.suffix != ".json" or not calibration_path.is_file():
    raise ValueError(
        f"Expected a follower calibration JSON file at {calibration_path}"
    )

config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id=calibration_path.stem,
    calibration_dir=calibration_path.parent,
    use_degrees=True,
    max_relative_target=10.0,
)

robot = SO101Follower(config)


# ==============================
# Relative P2P movement
# ==============================

def move_relative(offsets, duration=4.0, fps=30):

    start = get_positions(robot)

    target = {
        joint: position + offsets.get(joint, 0.0)
        for joint, position in start.items()
    }

    print("Starting position:", start)
    print("Target position:", target)

    try:
        for _, resume_at in move_steps(robot, start, offsets, duration, fps):
            time.sleep(max(0.0, resume_at - time.monotonic()))
    except TargetNotReached:
        print("Warning: Target not fully reached.")
        return False
    print("Target reached!")
    return True


# ==============================
# Main program
# ==============================

if __name__ == "__main__":

    robot.connect()

    try:

        move_relative({
            "shoulder_pan.pos": 20.0,
            "shoulder_lift.pos": 20.0,
            "elbow_flex.pos": -30.0,
            "wrist_flex.pos": 20.0,
            "wrist_roll.pos": 20.0,
            "gripper.pos": 2.0,
        }, duration=4.0)

        print("Final position:", get_positions(robot))

    finally:
        try:
            input("Support the arm and press Enter to disconnect.")
        finally:
            robot.disconnect()
