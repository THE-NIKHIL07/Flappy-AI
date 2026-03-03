import pygame
import random
import gymnasium as gym
from gymnasium import spaces
import os

class FlappyBirdEnv(gym.Env):
    def __init__(self, render_mode=None, frame_skip=1, headless=False):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.headless = headless
        self.width = 800
        self.height = 700
        self.screen = None
        self.clock = None

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=float)

        base_path = os.path.join(os.path.dirname(__file__), "../assets")
        self.bird_path = os.path.join(base_path, "bird.png")
        self.visual_scale = 1.2
        self.visual_bird_img = None

        self.collision_w = 34
        self.collision_h = 24

        self.flap_sound_path = os.path.join(base_path, "flap.mp3")
        self.score_sound_path = os.path.join(base_path, "score.mp3")
        self.die_sound_path = os.path.join(base_path, "die.mp3")

        self.flap_sound = None
        self.score_sound = None
        self.die_sound = None

        self.last_pipe_y = None
        self.pipe_gap = 130
        self.pipe_speed = 4
        self.gravity = 0.8
        self.flap_strength = -7
        self.tolerance = 10

    def _init_display(self):
        if self.headless:
            return
        if self.screen is None:
            pygame.init()
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24, bold=True)

            img = pygame.image.load(self.bird_path).convert_alpha()
            self.visual_bird_img = pygame.transform.scale(
                img, (int(self.collision_w * self.visual_scale), int(self.collision_h * self.visual_scale))
            )

            self.flap_sound = pygame.mixer.Sound(self.flap_sound_path)
            self.score_sound = pygame.mixer.Sound(self.score_sound_path)
            self.die_sound = pygame.mixer.Sound(self.die_sound_path)

            self.flap_sound.set_volume(0.7)
            self.score_sound.set_volume(0.9)
            self.die_sound.set_volume(0.9)

    def reset(self, seed=None):
        self.bird_y = 256
        self.bird_vel = 0
        self.pipe_x = 450
        self.pipe_y = random.randint(200, 300)
        self.last_pipe_y = self.pipe_y
        self.score = 0
        self.passed_pipe = False
        return self._get_state(), {}

    def step(self, action):
        total_reward = 0
        done = False

        for _ in range(self.frame_skip):
            if action == 1:
                self.bird_vel = self.flap_strength
                if self.flap_sound and hasattr(self.flap_sound, "play"):
                    self.flap_sound.play()

            self.bird_vel += self.gravity
            self.bird_y += self.bird_vel
            self.pipe_x -= self.pipe_speed

            reward = 0.2
            pipe_center = self.pipe_y + self.pipe_gap / 2
            reward += 0.01 * (50 - abs(self.bird_y - pipe_center))

            if not self.passed_pipe and self.pipe_x + 52 < 50:
                reward = 1.0
                self.score += 1
                self.passed_pipe = True
                if self.score_sound and hasattr(self.score_sound, "play"):
                    self.score_sound.play()

            if self.pipe_x < -52:
                self.pipe_x = 400
                if random.random() < 0.4:
                    change = random.randint(-170, 170)
                else:
                    change = random.randint(-90, 90)
                new_pipe_y = self.last_pipe_y + change
                self.pipe_y = max(80, min(420, new_pipe_y))
                self.last_pipe_y = self.pipe_y
                self.passed_pipe = False

            if (
                self.bird_y < 0 or
                self.bird_y > 400 or
                (
                    50 + self.collision_w > self.pipe_x and
                    50 < self.pipe_x + 52 and
                    (self.bird_y < self.pipe_y + self.tolerance or self.bird_y + self.collision_h > self.pipe_y + self.pipe_gap - self.tolerance)
                )
            ):
                reward = -1.0
                done = True
                if self.die_sound and hasattr(self.die_sound, "play"):
                    self.die_sound.play()

            total_reward += reward
            if done:
                break

        return self._get_state(), total_reward, done, False, {"score": self.score}

    def _get_state(self):
        pipe_center = self.pipe_y + self.pipe_gap / 2
        return [
            self.bird_y / self.height,
            self.bird_vel / 10,
            self.pipe_x / self.width,
            self.pipe_y / self.height,
            (self.pipe_y - self.bird_y) / self.height,
            (self.pipe_y + self.pipe_gap - self.bird_y) / self.height,
            (pipe_center - self.bird_y) / self.height
        ]

    def render(self, best_score=0):
        if self.headless:
            return
        self._init_display()
        self.screen.fill((135, 206, 235))
        pygame.draw.rect(self.screen, (0, 200, 0), (self.pipe_x, 0, 52, self.pipe_y))
        pygame.draw.rect(self.screen, (0, 200, 0), (self.pipe_x, self.pipe_y + self.pipe_gap, 52, 512))

        angle = -self.bird_vel * 3
        bird = pygame.transform.rotate(self.visual_bird_img, angle)
        visual_w, visual_h = bird.get_size()
        x_center = 50 + self.collision_w // 2
        y_center = max(visual_h // 2, min(self.height - visual_h // 2, self.bird_y + self.collision_h // 2))
        rect = bird.get_rect(center=(x_center, y_center))
        self.screen.blit(bird, rect.topleft)

        self.screen.blit(self.font.render(f"Score: {self.score}", True, (255, 255, 255)), (10, 10))
        self.screen.blit(self.font.render(f"Best: {best_score}", True, (255, 255, 255)), (10, 40))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if not self.headless:
            pygame.mixer.quit()
            pygame.display.quit()
            pygame.quit()