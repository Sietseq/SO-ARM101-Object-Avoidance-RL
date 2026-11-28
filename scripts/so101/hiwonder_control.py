"""Calibrated six-axis HX-30HM feedback and speed-limited position commands."""

import math
from struct import pack

from lerobot.motors.encoding_utils import decode_sign_magnitude
from lerobot.motors.hiwonder import HiwonderMotorsBus
from lerobot.motors.hiwonder.hiwonder_sdk import COMM_SUCCESS, GroupSyncRead


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'Expected a finite number, got {value!r}')
    return float(value)


class FollowerPositionReader(GroupSyncRead):
    def getData(self, motor_id, address, length):
        # The SDK can return COMM_SUCCESS when only the last servo replied.
        if not self.last_result or self.isAvailable(motor_id, address, length) != (True, 0):
            raise ConnectionError(f'Incomplete follower position feedback for servo {motor_id}')
        return super().getData(motor_id, address, length)


class ControlBus(HiwonderMotorsBus):
    # send_action applies its per-joint cap before calling sync_write.
    check_position_write = None

    def sync_write(self, data_name, values, *, normalize=True, num_retry=0):
        if data_name == 'Goal_Position' and self.check_position_write is not None:
            self.check_position_write(values)
        return super().sync_write(data_name, values, normalize=normalize, num_retry=num_retry)


class HiwonderArm:
    """Keep the motor IDs, register units and signed encoder mappings at the bus."""

    def __init__(self, bus, calibration, mapping, config):
        self.bus = bus
        bus.sync_reader = FollowerPositionReader(bus.packet_handler, 56, 2)
        self.names = list(calibration)
        self.mapping = mapping
        self.calibration = calibration
        self.limits = {}
        self.speed_ticks = {}
        if set(config['joints']) != set(self.names):
            raise ValueError('Motion limits must cover all six joints')
        for name in self.names:
            row = config['joints'][name]
            low, high, speed = (number(row[k]) for k in ('min_rad', 'max_rad', 'max_rad_s'))
            zero, home, scale = mapping[name]
            cal = calibration[name]
            endpoints = sorted(home + (cal[k] - zero) * scale for k in ('range_min', 'range_max'))
            if not endpoints[0] <= low <= home <= high <= endpoints[1] or low >= high:
                raise ValueError(f'Invalid physical limits for {name}')
            speed_tick = math.floor(speed / abs(scale))
            if not 1 <= speed_tick <= 32767:
                raise ValueError(f'Invalid HX-30HM speed limit for {name}')
            self.limits[name] = low, high, speed_tick * abs(scale)
            self.speed_ticks[name] = speed_tick
        self.commanded = {}
        self.engaged = False
        self.tracking_error = number(config['tracking_error_rad'])
        # One HX-30HM SYNC READ from Torque_Enable (40) through Moving (66).
        self.feedback = GroupSyncRead(bus.packet_handler, 40, 27)
        for name in self.names:
            if not self.feedback.addParam(calibration[name]['id']):
                raise ValueError(f'Duplicate feedback ID for {name}')

    def read(self, register, name):
        return self.bus.read(register, name, normalize=False)

    def ticks(self, positions):
        result = {}
        for name, pos in zip(self.names, positions, strict=True):
            pos = number(pos)
            low, high, _ = self.limits[name]
            zero, home, scale = self.mapping[name]
            if not low <= pos <= high:
                raise ValueError(f'{name} target {pos} outside [{low}, {high}]')
            tick = round(zero + (pos - home) / scale)
            # Round inward at a physical limit.
            if home + (tick - zero) * scale < low:
                tick += 1 if scale > 0 else -1
            elif home + (tick - zero) * scale > high:
                tick -= 1 if scale > 0 else -1
            result[name] = tick
        return result

    def joint_feedback(self, name, tick, speed_ticks, *, enforce_speed=True):
        zero, home, scale = self.mapping[name]
        position = home + (tick - zero) * scale
        velocity = speed_ticks * scale
        low, high, speed = self.limits[name]
        if not low <= position <= high or (enforce_speed and abs(velocity) > speed):
            raise ValueError(f'{name} outside motion limits: {position} rad, {velocity} rad/s')
        return position, velocity

    def follower_target(self, positions):
        ticks = self.ticks(positions)
        ids = {self.calibration[n]['id']: tick for n, tick in ticks.items()}
        values = self.bus._normalize(ids)
        # The fork truncates on write. Avoid losing a tick to floating-point roundoff.
        for motor_id, raw in self.bus._unnormalize(values).items():
            if raw != ids[motor_id]:
                values.update(self.bus._normalize({motor_id: ids[motor_id] + .25}))
        return {n + '.pos': values[self.calibration[n]['id']] for n in self.names}

    def action_ticks(self, action):
        values = {self.calibration[n]['id']: action[n + '.pos'] for n in self.names}
        raw = self.bus._unnormalize(values)
        return {n: raw[self.calibration[n]['id']] for n in self.names}

    def tick_positions(self, ticks):
        return [self.joint_feedback(n, ticks[n], 0)[0] for n in self.names]

    def last_follower_positions(self):
        ticks = {n: self.bus.sync_reader.getData(self.calibration[n]['id'], 56, 2)
                 for n in self.names}
        return self.tick_positions(ticks)

    def sample(self, *, enforce_speed=True):
        group = self.feedback
        result = group.txRxPacket()
        if result != COMM_SUCCESS or not group.last_result:
            raise ConnectionError(f'Incomplete HX-30HM group feedback (SDK result {result})')
        positions, velocities = [], []
        moving = False
        for name in self.names:
            motor_id = self.calibration[name]['id']
            available, error = group.isAvailable(motor_id, 40, 27)
            if not available or error or group.getData(motor_id, 65, 1):
                raise RuntimeError(f'{name} missing feedback or reporting a servo fault')
            tick = decode_sign_magnitude(group.getData(motor_id, 56, 2), 15)
            speed = decode_sign_magnitude(group.getData(motor_id, 58, 2), 15)
            position, velocity = self.joint_feedback(name, tick, speed, enforce_speed=enforce_speed)
            if self.engaged:
                if (group.getData(motor_id, 40, 1) != 1
                        or group.getData(motor_id, 42, 2) != self.commanded[name]
                        or group.getData(motor_id, 46, 2) != self.speed_ticks[name]):
                    raise RuntimeError(f'{name} did not retain commanded torque, target or speed')
            moving |= group.getData(motor_id, 66, 1) != 0
            positions.append(position)
            velocities.append(velocity)
        return positions, velocities, moving

    def send(self, ticks):
        # Vendor SYNC WRITE, address 42: position, time=0, speed in steps/s.
        from lerobot.motors.hiwonder.hiwonder_sdk import COMM_SUCCESS, GroupSyncWrite
        writer = GroupSyncWrite(self.bus.packet_handler, 42, 6)
        for name, tick in ticks.items():
            data = pack('<HHH', tick, 0, self.speed_ticks[name])
            if not writer.addParam(self.calibration[name]['id'], data):
                raise RuntimeError(f'Cannot encode target for {name}')
        if writer.txPacket() != COMM_SUCCESS:
            raise ConnectionError('HX-30HM group position write failed')
        self.commanded = ticks.copy()

    def engage(self, positions, *, enforce_speed=True):
        ticks = self.ticks(positions)
        for name, tick in ticks.items():
            if (self.read('Operating_Mode', name) != 0 or self.read('Moving', name) != 0
                    or self.read('Present_Position', name) != tick
                    or self.read('Torque_Enable', name) not in (0, 1)):
                raise RuntimeError(f'{name} changed before its initial hold')
        self.engaged = True  # A group write can engage torque even if the host loses the reply.
        self.send(ticks)
        for name, tick in ticks.items():
            goal = self.read('Goal_Position', name)
            speed = self.read('Goal_Velocity', name)
            moving = self.read('Moving', name)
            position = self.read('Present_Position', name)
            _, velocity = self.joint_feedback(
                name, position, self.read('Present_Velocity', name), enforce_speed=enforce_speed)
            expected_speed = self.speed_ticks[name]
            torque = self.read('Torque_Enable', name)
            # A powered joint can settle under supervision. An unpowered joint
            # must still be at rest at the captured target before enabling torque.
            error = abs((position - tick) * self.mapping[name][2])
            if (goal != tick or speed != expected_speed or error > self.tracking_error
                    or (torque == 0 and (moving or velocity or position != tick))):
                raise RuntimeError(
                    f'{name} failed initial hold readback. Requested position {tick} ticks, '
                    f'Goal_Position={goal}, Present_Position={position}, '
                    f'Goal_Velocity={speed} (expected {expected_speed} ticks/s), '
                    f'Moving={moving}, Torque_Enable={torque}, velocity={velocity:.5f} rad/s')
            if torque == 0:
                self.bus.write('Torque_Enable', name, 1, normalize=False)
            elif torque != 1:
                raise RuntimeError(f'Unexpected torque mode {torque} on {name}')

    def stop(self):
        """Hold each reachable powered axis at a fresh measurement. Preserve gravity support."""
        if not self.engaged:
            return
        failures = []
        for name in self.names:
            try:
                torque = self.read('Torque_Enable', name)
                if torque == 0:
                    continue
                if torque != 1 or self.read('Operating_Mode', name) != 0:
                    raise RuntimeError('position control unavailable')
                tick = self.read('Present_Position', name)
                if not 0 <= tick <= 4095:
                    raise ValueError(f'invalid encoder count {tick}')
                self.bus.write('Goal_Position', name, tick, normalize=False)
                if self.read('Goal_Position', name) != tick:
                    raise RuntimeError('hold target did not latch')
                self.commanded[name] = tick
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append(f'{name}: {exc}')
        if failures:
            raise RuntimeError('Hold unconfirmed on ' + ', '.join(failures))
