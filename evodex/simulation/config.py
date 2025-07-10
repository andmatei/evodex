import math

DEFAULT_ROBOT_CONFIG = {
    "base": {
        "width": 30,
        "height": 100,
    },
    "fingers": [
        {
            "num_segments": 3,
            "segment_lengths": [50, 40, 30],
            "segment_widths": [30, 25, 20],
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
}

DEFAULT_SCENARIO_CONFIG = {
    "type": "MoveCubeToTargetScenario",  # Default scenario type
    "target_position": (600, 300),  # Default target position for scenarios
}

DEFAULT_SIMULATOR_CONFIG = {
    "simulation": {
        "dt": 1.0 / 60.0,
        "gravity": (0, 900),
        "screen_width": 800,
        "screen_height": 600,
        "world_scale": 1.0,
    },
    "render": {
        "enabled": True,
        "fps": 60,
        "draw_options": {
            "draw_fps": True,
            "draw_collision_shapes": False,
            "draw_constraints": True,
            "draw_kinematics": True,
        },
    },
    "logging": {
        "enabled": True,
        "log_level": "INFO",  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        "log_file": "simulation.log",
    },
    "keyboard_control": {
        "enabled": True,
        "move_speed": 150,  # Speed for keyboard-controlled base movement (pixels/sec)
        "angular_speed": 1.5,  # Angular speed for keyboard-controlled base rotation (radians/sec)
    },
    "collision": {
        "ground": 0,
        "robot_base": 1,
        "robot_segment_start": 3,
        "scenario_object_start": 100,
        "scenario_static_start": 200,
    },
    "pymunk_to_pygame_coord": {
        "scale": 1.0,  # Scale factor for converting Pymunk coordinates to Pygame coordinates
    },
    "pygame_to_pymunk_coord": {
        "scale": 1.0,  # Scale factor for converting Pygame coordinates to Pymunk coordinates
    },
}
