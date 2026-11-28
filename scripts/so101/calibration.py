"""Load calibrated HX-30HM joint mappings and verify the connected follower."""

from contextlib import contextmanager
from hashlib import sha256
import json
import math


def load_calibration(path, names, max_tick):
    """Load follower IDs and the programmed position limits and offsets."""
    contents = path.read_bytes()
    records = json.loads(contents)
    if not isinstance(records, dict) or set(records) != set(names):
        raise ValueError('Follower calibration must contain the six training joint names')
    ids = set()
    for name, row in records.items():
        fields = ('id', 'drive_mode', 'homing_offset', 'range_min', 'range_max')
        if not isinstance(row, dict) or any(type(row.get(k)) is not int for k in fields):
            raise ValueError(f'Invalid follower calibration fields for {name}')
        if not (0 <= row['id'] <= 252 and row['id'] not in ids and row['drive_mode'] in (0, 1)
                and 0 <= row['range_min'] < row['range_max'] <= max_tick):
            raise ValueError(f'Invalid motor ID, drive mode or calibrated range for {name}')
        ids.add(row['id'])
    return records, sha256(contents).hexdigest()


def load_mapping(path, digest, calibration, home, radians_per_tick):
    """Load signed encoder scales from a model-zero pose or two angular references."""
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['calibration_sha256'] != digest:
        raise ValueError('Joint mapping belongs to a different calibration file')
    if set(data['joints']) != set(home):
        raise ValueError('Joint mapping must cover all six joints')
    mapping = {}
    if 'reference_pose' in data:
        if data['reference_pose'] != 'model_zero':
            raise ValueError('Unknown reference pose in joint mapping')
        for name, home_rad in home.items():
            row = data['joints'][name]
            zero, sign = row['zero_tick'], row['sign']
            low, high = calibration[name]['range_min'], calibration[name]['range_max']
            if type(zero) is not int or not low <= zero <= high:
                raise ValueError(f'Invalid model-zero encoder count for {name}')
            if type(sign) is not int or sign not in (-1, 1):
                raise ValueError(f'Invalid measured direction for {name}')
            scale = sign * radians_per_tick
            if not low <= zero + home_rad / scale <= high:
                raise ValueError(f'Training home outside calibrated range for {name}')
            mapping[name] = (zero, 0.0, scale)
        return mapping
    tolerance = data['tolerance_rad']
    if type(tolerance) not in (float, int) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Mapping tolerance_rad must be a finite positive measurement tolerance')
    for name, home_rad in home.items():
        row = data['joints'][name]
        baseline, check, angle = row['home_tick'], row['check_tick'], row['check_rad']
        limits = calibration[name]
        if not all(type(t) is int and limits['range_min'] <= t <= limits['range_max']
                   for t in (baseline, check)):
            raise ValueError(f'Reference ticks outside the calibrated interval for {name}')
        if type(angle) not in (int, float) or not math.isfinite(angle):
            raise ValueError(f'Invalid independently measured angle for {name}')
        displacement = angle - home_rad
        rotation = (check - baseline) * radians_per_tick
        if (abs(displacement) <= 2 * tolerance
                or abs(abs(rotation) - abs(displacement)) > tolerance):
            raise ValueError(f'Second reference fails the encoder scale check for {name}')
        signed_scale = math.copysign(radians_per_tick, rotation * displacement)
        mapping[name] = (baseline, home_rad, signed_scale)
    return mapping


@contextmanager
def connected_reader(bus, calibration, baudrate=None, *, exclusive=False):
    """Verify each responding servo's registers and preserve torque on every exit."""
    try:
        # Check individual IDs via their programmed registers after selecting the host baud rate.
        bus.connect(handshake=False)
        if baudrate is not None and bus.get_baudrate() != baudrate:
            bus.set_baudrate(baudrate)  # Host serial port only.
        if exclusive:
            bus.port_handler.ser.exclusive = True
        actual = bus.read_calibration()
        for name, row in calibration.items():
            fields = ('id', 'homing_offset', 'range_min', 'range_max')
            if name not in actual or any(getattr(actual[name], k) != row[k] for k in fields):
                raise ValueError(f'Servo calibration mismatch for {name} (ID {row["id"]})')
        yield
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
