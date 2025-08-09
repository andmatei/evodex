# --- Collision Types ---
COLLISION_TYPE_GROUND = 0
COLLISION_TYPE_SCENARIO_OBJECT_START = 100
COLLISION_TYPE_SCENARIO_STATIC_START = 200


def pymunk_to_pygame_coord(point, height, scale=1.0):
    return int(point[0] * scale), int((height - point[1]) * scale)


def pygame_to_pymunk_coord(point, height, scale=1.0):
    return float(point[0] / scale), float((height - point[1]) / scale)
