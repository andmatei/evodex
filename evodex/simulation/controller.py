import pygame

from .config import KeyboardControlConfig
from .robot import Action


class KeyboardController:
    def __init__(self, config: KeyboardControlConfig):
        self.config = config

        self.angular_mode = False

        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_omega = 0.0

    def handle_event(self, event: pygame.event.Event):
        if self.config.enabled is False:
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.angular_mode = True
                self.target_vx = 0.0

                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    self.target_omega = -self.config.angular_speed
                elif keys[pygame.K_RIGHT]:
                    self.target_omega = self.config.angular_speed
            elif event.key == pygame.K_UP:
                self.target_vy = -self.config.move_speed
            elif event.key == pygame.K_DOWN:
                self.target_vy = self.config.move_speed
            elif event.key == pygame.K_LEFT:
                if self.angular_mode:
                    self.target_omega = -self.config.angular_speed
                    self.target_vx = 0.0
                else:
                    self.target_vx = -self.config.move_speed
                    self.target_omega = 0.0
            elif event.key == pygame.K_RIGHT:
                if self.angular_mode:
                    self.target_omega = self.config.angular_speed
                    self.target_vx = 0.0
                else:
                    self.target_vx = self.config.move_speed
                    self.target_omega = 0.0

        elif event.type == pygame.KEYUP:
            if (event.key == pygame.K_UP and self.target_vy < 0) or (
                event.key == pygame.K_DOWN and self.target_vy > 0
            ):
                self.target_vy = 0.0
            elif event.key == pygame.K_SPACE:
                self.angular_mode = False
                self.target_omega = 0.0

                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    self.target_vx = -self.config.move_speed
                elif keys[pygame.K_RIGHT]:
                    self.target_vx = self.config.move_speed
            if self.angular_mode:
                if (event.key == pygame.K_LEFT and self.target_omega < 0) or (
                    event.key == pygame.K_RIGHT and self.target_omega > 0
                ):
                    self.target_omega = 0.0
            else:
                if (event.key == pygame.K_LEFT and self.target_vx < 0) or (
                    event.key == pygame.K_RIGHT and self.target_vx > 0
                ):
                    self.target_vx = 0.0

    def get_action(self):
        return {
            "base": {
                "velocity": (self.target_vx, self.target_vy),
                "omega": self.target_omega,
            },
        }
