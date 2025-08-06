import pygame


class ManualController:
    def __init__(self, config: dict):
        self.config = config

        self.key_move_speed = self.config.get("key_move_speed", 150)
        self.key_angular_speed = self.config.get("key_angular_speed", 1.5)

        self.angular_mode = False

        self.base_target_vx = 0.0
        self.base_target_vy = 0.0
        self.base_target_omega = 0.0

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.angular_mode = True
                self.base_target_vx = 0.0

                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    self.base_target_omega = -self.key_angular_speed
                elif keys[pygame.K_RIGHT]:
                    self.base_target_omega = self.key_angular_speed
            elif event.key == pygame.K_UP:
                self.base_target_vy = -self.key_move_speed
            elif event.key == pygame.K_DOWN:
                self.base_target_vy = self.key_move_speed
            elif event.key == pygame.K_LEFT:
                if self.angular_mode:
                    self.base_target_omega = -self.key_angular_speed
                    self.base_target_vx = 0.0
                else:
                    self.base_target_vx = -self.key_move_speed
                    self.base_target_omega = 0.0
            elif event.key == pygame.K_RIGHT:
                if self.angular_mode:
                    self.base_target_omega = self.key_angular_speed
                    self.base_target_vx = 0.0
                else:
                    self.base_target_vx = self.key_move_speed
                    self.base_target_omega = 0.0

        elif event.type == pygame.KEYUP:
            if (event.key == pygame.K_UP and self.base_target_vy < 0) or (
                event.key == pygame.K_DOWN and self.base_target_vy > 0
            ):
                self.base_target_vy = 0.0
            elif event.key == pygame.K_SPACE:
                self.angular_mode = False
                self.base_target_omega = 0.0

                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    self.base_target_vx = -self.key_move_speed
                elif keys[pygame.K_RIGHT]:
                    self.base_target_vx = self.key_move_speed
            if self.angular_mode:
                if (event.key == pygame.K_LEFT and self.base_target_omega < 0) or (
                    event.key == pygame.K_RIGHT and self.base_target_omega > 0
                ):
                    self.base_target_omega = 0.0
            else:
                if (event.key == pygame.K_LEFT and self.base_target_vx < 0) or (
                    event.key == pygame.K_RIGHT and self.base_target_vx > 0
                ):
                    self.base_target_vx = 0.0

    def get_actions(self):
        return {
            "base_vx": self.base_target_vx,
            "base_vy": self.base_target_vy,
            "base_omega": self.base_target_omega,
        }
