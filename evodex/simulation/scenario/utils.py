from evodex.simulation.robot.spaces import ExtrinsicObservation
from .types import ObjectObservation

# --- Collision Types ---
COLLISION_TYPE_GROUND = 0
COLLISION_TYPE_GRASPING_OBJECT = 1


def pymunk_to_pygame_coord(point, height, scale=1.0):
    return int(point[0] * scale), int((height - point[1]) * scale)


def pygame_to_pymunk_coord(point, height, scale=1.0):
    return float(point[0] / scale), float((height - point[1]) / scale)


def get_relative_observation(
    robot_obs: ExtrinsicObservation, object_obs: ObjectObservation
) -> Observation:
    return Observation(
        velocity=tuple(np.array(robot_obs.velocity) - np.array(object_pos)),
        position=tuple(np.array(robot_obs.position) - np.array(object_pos)),
        angle=robot_obs.angle,
        angular_velocity=robot_obs.angular_velocity,
        base_to_obj=tuple(np.array(object_pos) - np.array(robot_obs.base.position)),
        fingertips_to_obj=[
            tuple(np.array(object_pos) - finger.position)
            for finger in robot_obs.fingertips
        ],
    )
