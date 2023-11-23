# Read motor configurations
def read_motor_config(filename):
    motors = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split(',')
            x, y, orientation = map(int, parts)
            motors.append([x, y, orientation, False])  # Adding False for motor activation status
    return motors

colors = {
    'BLACK': (0, 0, 0),
    'BLUE': (100, 100, 255),
    'RED': (255, 0, 0),
    'WHITE': (255, 255, 255),
    'GREEN': (0, 255, 0),
}