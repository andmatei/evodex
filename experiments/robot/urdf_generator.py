import yaml
from jinja2 import Environment, FileSystemLoader
import math

# --- Helper functions for pre-processing ---

def calculate_cylinder_inertia(mass, radius, depth):
    """Calculates the inertia tensor for a cylinder oriented along the Z-axis."""
    ixx = (1/12) * mass * (3 * radius**2 + depth**2)
    iyy = ixx
    izz = 0.5 * mass * radius**2
    return {'ixx': ixx, 'ixy': 0, 'ixz': 0, 'iyy': iyy, 'iyz': 0, 'izz': izz}

def calculate_capsule_inertia(mass, length, radius):
    """Calculates a simplified inertia tensor for a capsule oriented along the X-axis."""
    # This is an approximation treating it similarly to a cylinder along X
    ixx = 0.5 * mass * radius**2
    iyy = (1/12) * mass * (3 * radius**2 + length**2)
    izz = iyy
    return {'ixx': ixx, 'ixy': 0, 'ixz': 0, 'iyy': iyy, 'iyz': 0, 'izz': izz}

def calculate_sphere_inertia(mass, radius):
    """Calculates the inertia tensor for a sphere."""
    I = 0.4 * mass * radius**2
    return {'ixx': I, 'ixy': 0, 'ixz': 0, 'iyy': I, 'iyz': 0, 'izz': I}


def preprocess_config(config):
    """Adds calculated values needed by the Jinja template."""
    robot_data = config

    # --- FIX: Calculate inertia for the base ---
    base = robot_data['base']
    if base['visual']['type'] == 'cylinder':
        base['inertia'] = calculate_cylinder_inertia(
            base['mass'], base['visual']['radius'], base['visual']['depth']
        )
    # Add logic here for other base shapes like 'sphere' or 'box' if needed.

    # --- Process Fingers ---
    for finger in robot_data.get('fingers', []):
        # Convert attachment angle to radians
        if 'attachment' in finger and 'angle_deg' in finger['attachment']:
            finger['attachment']['angle_rad'] = math.radians(finger['attachment']['angle_deg'])

        all_parts = finger.get('segments', []) + ([finger.fingertip] if 'fingertip' in finger else [])
        for part in all_parts:
            # Pre-calculate inertia tensor based on shape
            if part['visual']['type'] == 'capsule':
                part['inertia'] = calculate_capsule_inertia(
                    part['mass'], part['visual']['length'], part['visual']['radius']
                )
            elif part['visual']['type'] == 'sphere':
                 part['inertia'] = calculate_sphere_inertia(
                    part['mass'], part['visual']['radius']
                )
            # Add logic for other segment shapes if needed

    return robot_data

# --- Main script logic ---
# Load the configuration
with open("./configs/robot/3d/base_robot.yaml", 'r') as f:
    config_data = yaml.safe_load(f)

# Pre-process the data
processed_config = preprocess_config(config_data)

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader(searchpath="./evodex/templates"))
template = env.get_template("robot.urdf.j2")

# Render the template
urdf_content = template.render(
    robot=processed_config,
    cos=math.cos,
    sin=math.sin
)

# Save the output
with open("generated_gripper.urdf", "w") as f:
    f.write(urdf_content)

print("URDF file 'generated_gripper.urdf' has been created.")