import pygame
import pymunk


class Simulation:

    def __init__(self, scenario, robot, mode = 'agent'):
        self.scenario = scenario
        self.robot = robot
        self.mode = mode
        
        self.space = pymunk.Space()
        self.space.gravity = (0, -9.81)
        self.scenario.setup(self.space)
        self.robot.setup(self.space)

        self.clock = pygame.time.Clock()

        pygame.init()

        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Simulation")

    

