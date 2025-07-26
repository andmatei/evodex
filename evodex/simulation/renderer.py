import pygame
import pymunk
import pymunk.pygame_util

from .scenario import Scenario
from .config import RenderConfig


class Renderer:
    def __init__(self, width: int, height: int, config: RenderConfig):
        self.config = config

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        pygame.display.set_caption("Robot Hand Simulation")

    def render(self, space: pymunk.Space, scenario: Scenario):
        self.screen.fill(pygame.Color("white"))
        space.debug_draw(self.draw_options)

        scenario.render(self.screen)

        pygame.display.flip()
        self.clock.tick(self.config.fps)

        if self.config.draw_options.draw_fps:
            fps = self.clock.get_fps()
            font = pygame.font.Font(None, 36)
            text = font.render(f"FPS: {fps:.2f}", True, pygame.Color("black"))
            self.screen.blit(text, (10, 10))

    def close(self):
        pygame.quit()
