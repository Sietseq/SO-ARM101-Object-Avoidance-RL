"""The follower's point-to-point move for both standalone and ROS callers."""

from math import ceil
from time import monotonic


class TargetNotReached(RuntimeError):
    """The follower did not reach its endpoint within the retry period."""


def get_positions(robot):
    return {k: v for k, v in robot.get_observation().items() if k.endswith('.pos')}


def move_relative(robot, start, offsets, duration=4.0, fps=30):
    """Yield each sent command and the earliest time to resume the move."""
    if offsets.keys() - start.keys():
        raise ValueError(f'Unknown joints: {offsets.keys() - start.keys()}')
    target = {k: value + offsets.get(k, 0.0) for k, value in start.items()}
    if not 0 <= target['gripper.pos'] <= 100:
        raise ValueError('Gripper target outside 0-100')
    steps = max(1, ceil(duration * fps))
    for i in range(1, steps + 1):
        t = i / steps
        s = 3*t**2 - 2*t**3
        action = {k: value + s * (target[k] - value) for k, value in start.items()}
        yield robot.send_action(action), monotonic() + 1 / fps

    deadline = monotonic() + 3.0
    while monotonic() < deadline:
        current = get_positions(robot)
        if max(abs(value - current[k]) for k, value in target.items()) <= 1.0:
            return
        yield robot.send_action(target), monotonic() + 1 / fps
    raise TargetNotReached('Follower did not reach the target within 3 seconds')
