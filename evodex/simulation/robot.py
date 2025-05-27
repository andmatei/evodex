import pymunk
import numpy as np

class Segment:
    def __init__(self, space, position, length, width, angle=0.0, is_fingertip=False, mass=1.0):
        self.length = length
        self.width = width
        self.is_fingertip = is_fingertip
        moment = pymunk.moment_for_box(mass, (length, width))
        self.body = pymunk.Body(mass, moment)
        self.body.position = position
        self.body.angle = angle

        self.shape = pymunk.Poly.create_box(self.body, (length, width))
        self.shape.mass = mass
        self.shape.friction = 0.8
        self.shape.elasticity = 0.3

        space.add(self.body, self.shape)

    def get_tip_position(self):
        """
        Returns the position of the tip of the segment.
        """
        local_tip = (self.length / 2, 0)
        return self.body.local_to_world(local_tip)
    


class Finger:
    def __init__(self, space, base, attach_point, config):
        self.space = space
        self.segments = []
        self.joints = []
        self.motors = []

        self.config = config
        self.num_segments = config['num_segments']

        prev_body = base
        current_attach_point = base.local_to_world(attach_point)

        for i in range(self.num_segments):
            segment_length = config['segment_lengths'][i]
            segment_width = config['segment_widths'][i]
            is_fingertip = (i == self.num_segments - 1)

            if i == 0:
                initial_angle = base.angle
                segment_pos_x = current_attach_point[0] + (segment_length / 2) * np.cos(initial_angle)
                segment_pos_y = current_attach_point[1] + (segment_length / 2) * np.sin(initial_angle)
            else:
                prev_segment_tip = self.segments[-1].get_tip_position()
                initial_angle = self.segments[-1].body.angle
                segment_pos_x = prev_segment_tip[0] + (segment_length / 2) * np.cos(prev_body.angle)
                segment_pos_y = prev_segment_tip[1] + (segment_length / 2) * np.sin(prev_body.angle)

            segment = Segment(space, (segment_pos_x, segment_pos_y), segment_length, segment_width, initial_angle, is_fingertip)
            self.segments.append(segment)

            joint_pos_world = current_attach_point if i == 0 else self.segments[-2].get_tip_position()
            anchor_a = prev_body.world_to_local(joint_pos_world)
            anchor_b = segment.body.local_to_world(joint_pos_world)
            joint = pymunk.PinJoint(prev_body, segment.body, anchor_a, anchor_b)
            self.joints.append(joint)
            self.space.add(joint)

            motor = pymunk.SimpleMotor(prev_body, segment.body, 0.0)
            motor.max_force = 5e7
            self.motors.append(motor)
            self.space.add(motor)

            prev_body = segment.body
            current_attach_point = segment.get_tip_position()

        def set_motor_torque(self, segment_index, torque):
            """
            Set the torque for a specific segment motor.
            """
            if 0 <= segment_index < len(self.motors):
                self.motors[segment_index].rate = torque
            else:
                raise IndexError("Segment index out of range.")

        def set_motor_torques(self, torques):
            """
            Set the torques for all segment motors.
            """
            if len(torques) != len(self.motors):
                raise ValueError("Length of torques must match number of motors.")
            for motor, torque in zip(self.motors, torques):
                motor.rate = torque

        def get_joint_angles(self):
            angles = []
            for i in range(len(self.joints)):
                angle = self.segments[i].body.angle - self.motors[i].a.angle if i == 0 else self.segments[i].body.angle - self.segments[i-1].body.angle
                angles.append(angle) 
            return angles
        
        def get_tip_position(self):
            return self.segments[-1].get_tip_position() if self.segments else None
        
        def get_observation(self):
            """
            Get the current observation of the finger.
            This can include joint angles, positions, and other relevant data.
            """
            observation = []
            for segment in self.segments:
                observation.append(segment.body.position.x)
                observation.append(segment.body.position.y)
                observation.append(segment.body.angle)
                observation.append(segment.body.angular_velocity)
            return np.array(observation, dtype=np.float32)
        

class Base:
    def __init__(self, space, position, width, height, num_fingers):
        self.width = width
        self.height = height

        self.body = pymunk.Body(bpdy_type=pymunk.Body.KINEMATIC)
        self.body.position = position
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0.0

        self.shape = pymunk.Poly.create_box(self.body, (width, height))
        self.shape.friction = 0.9
        self.shape.elasticity = 0.2
        space.add(self.body, self.shape)

        self.finger_attachement_points = []
        for i in range(num_fingers):
            local_x = self.width / 2
            vertical_spacing = self.height / (num_fingers + 1)
            local_y = (i + 1) * vertical_spacing - self.height / 2
            self.finger_attachement_points.append((local_x, local_y))


class Robot:
    def __init__(self, space, config):
        self.space = space
        self.config = config

        num_fingers = len(config['fingers'])
        config['base']['num_fingers'] = num_fingers

        self.base = Base(space, config['base'])
        self.fingers = []

        for i, finger_config in enumerate(config['fingers']):
            attach_point = self.base.finger_attachement_points[i]
            finger = Finger(space, self.base.body, attach_point, finger_config)
            self.fingers.append(finger)

        self.num_motors = sum(finger.num_segments for finger in self.fingers)

    def get_dof(self):
        """
        Returns the total number of degrees of freedom (DOF) for the robot.
        This is the sum of all segments across all fingers.
        """
        return self.num_motors


    def apply_actions(self, actions):
        """
        Apply actions to the robot's fingers.
        Actions should be a list of torques for each finger segment.
        """
        if len(actions) != len(self.fingers):
            raise ValueError("Actions length must match number of fingers.")
        
        for finger, torques in zip(self.fingers, actions):
            finger.set_motor_torques(torques)

    def get_observation(self):
        """
        Get the current observation of the robot.
        This can include joint angles, positions, and other relevant data.
        """
        observation = [
            self.base.body.position.x,
            self.base.body.position.y,
            self.base.body.angle,
            self.base.body.angular_velocity,
            self.base.body.velocity.x,
            self.base.body.velocity.y
        ]
        for finger in self.fingers:
            observation.extend(finger.get_observation())
        return np.array(observation, dtype=np.float32)