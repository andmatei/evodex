from typing import Tuple

# --- Collision Types ---
COLLISION_TYPE_GROUND = 0
COLLISION_TYPE_GRASPING_OBJECT = 1


def pymunk_to_pygame_coord(
    point: Tuple[float, float], height: float, scale: float = 1.0
) -> Tuple[int, int]:
    return int(point[0] * scale), int((height - point[1]) * scale)


def pygame_to_pymunk_coord(
    point: Tuple[float, float], height: float, scale: float = 1.0
) -> Tuple[float, float]:
    return float(point[0] / scale), float((height - point[1]) / scale)
