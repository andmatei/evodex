import math

DEFAULT_ROBOT_CONFIG = {
    "base": {
        "width": 30,
        "height": 100,
        "kinematic": True,
        "initial_position": (250, 200),  # Centered a bit more for keyboard control
    },
    "fingers": [
        {
            "num_segments": 3,
            "segment_length": 40,
            "segment_width": 30,
            "motor_stiffness": 1e7,
            "motor_damping": 1e5,
            "joint_angle_limit_min": -math.pi / 4,
            "joint_angle_limit_max": math.pi / 4,
            "fingertip_shape": "circle",
            "fingertip_radius": 7,
            "fingertip_size": (10, 5),
            "motor_max_force": 5e7,
        },
        {
            "num_segments": 1,
            "segment_length": 50,
            "segment_width": 12,
            "motor_stiffness": 1e7,
            "motor_damping": 1e5,
            "joint_angle_limit_min": -math.pi / 3,
            "joint_angle_limit_max": math.pi / 3,
            "fingertip_shape": "rectangle",
            "fingertip_radius": 6,
            "fingertip_size": (12, 6),
            "motor_max_force": 5e7,
        },
    ],
    "simulation": {
        "dt": 1.0 / 60.0,
        "gravity": (0, 900),
        "screen_width": 800,
        "screen_height": 600,
        "world_scale": 1.0,
        "key_move_speed": 150,  # Speed for keyboard-controlled base movement (pixels/sec)
    },
}
