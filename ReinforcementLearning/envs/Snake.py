from gym import Env
from gym.spaces import Discrete, Box
from gym import core, spaces
from gym.error import DependencyNotInstalled
import numpy as np 

class SnakeWorldEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    SCREEN_SCALE = 30.0
    SCREEN_DIM = 28 * SCREEN_SCALE

    def __init__(self):
        self.screen = None
        self.clock = None
        self.isopen = True

        world_size = (28, 28)
        # Actions we can take, down, stay, up
        self.action_space = Discrete(4)
        self.observation_space = Box(low=-1, high=1, shape=(2, 1), dtype=np.float32)

        # World Space
        self.state = np.zeros(shape=world_size).astype(dtype='float32')

        self.snake_length = 3
        self.snake_head_position = (0, 0)
        self.apple_position = (0, 0)
        self.last_time_grown = 0
        self.snake_body = []

    def step(self, action):

        done = False
        reward = 0
        apple_consumed = False

        current_distance_from_apple = np.linalg.norm(self.apple_position - self.snake_head_position)
        # Update params.
        # Apply action
        # Remove head from previous.
        self.state[int(self.snake_head_position[0]), int(self.snake_head_position[1])] = 0
        # Move the last sequence to the next position.
        if action == 0:
            self.snake_head_position += np.array([1, 0])
        if action == 1:
            self.snake_head_position += np.array([-1, 0])
        if action == 2:
            self.snake_head_position += np.array([0, 1])
        if action == 3:
            self.snake_head_position += np.array([0, -1])

        if self.snake_head_position[0] >= self.state.shape[0] or self.snake_head_position[0] < 0:
            done = True
            reward = 0
        elif self.snake_head_position[1] >= self.state.shape[1] or self.snake_head_position[1] < 0:
            done = True
            reward = 0

        if not done:
            # Update new position
            self.state[int(self.snake_head_position[0]), int(self.snake_head_position[1])] = 255

            #
            self.last_time_grown += 1

            # If snake overlap.

            next_distance_from_apple = np.linalg.norm(self.apple_position - self.snake_head_position)

            reward = (1.0 / max(1.0, np.linalg.norm(self.apple_position - self.snake_head_position))) * 0.25
            # Reward negative going away.
            if next_distance_from_apple > current_distance_from_apple:
                reward = -reward

            if self.state[self.apple_position[0], self.apple_position[1]] == 255:
                apple_consumed = True
            # Calculate reward
            # If closer. small reward, get apple. big reward.
            if apple_consumed:
                reward += 20
                self.snake_length += 1
                self.last_time_grown = 0

                # Update apple position
                self.state[self.apple_position[0], self.apple_position[1]] = 0
                self.apple_position = np.random.randint(size=(2,), low=0, high=28)
                self.state[self.apple_position[0], self.apple_position[1]] = -255
            else:
                # Check if timeout.
                if self.last_time_grown >= 500 - self.snake_length:
                    done = True
                    reward = 0
                else:
                    done = False

        # Set placeholder for info
        info = {}

        Appledir = (self.apple_position - self.snake_head_position)
        distance = np.linalg.norm(Appledir)
        if distance > 0:
            Appledir = Appledir / np.linalg.norm(Appledir)
        else:
            Appledir = np.array([0, 0])

        # Return step information
        return Appledir, reward, done, info

    def reset(self):
        self.snake_length = 3
        self.snake_head_position = np.asarray(self.state.shape, dtype=int) / 2
        self.snake_last_position = np.asarray(self.snake_head_position - (0, 3))

        self.state = np.zeros(shape=self.state.shape).astype(dtype='float32')

#        for x in range(0, self.snake_length):
        self.state[int(self.snake_head_position[0]), int(self.snake_head_position[1])] = 255

        self.apple_position = np.random.randint(size=(2,), low=0, high=28)
        self.state[self.apple_position[0], self.apple_position[1]] = -255
        self.last_time_grown = 0

        Appledir = (self.apple_position - self.snake_head_position)
        distance = np.linalg.norm(Appledir)
        if distance > 0:
            Appledir = Appledir / np.linalg.norm(Appledir)
        else:
            Appledir = np.array([0, 0])

        #result = np.squeeze(np.array(self.state.flatten(), dtype=np.float32))

        return np.array([0, 0])

    def render(self, mode="human"):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        self.surf.fill((128, 128, 128))
        s = self.state

        # pygame.draw.
        pygame.draw.circle(self.surf, (0, 255, 0), np.asarray(self.apple_position) * self.SCREEN_SCALE, self.SCREEN_SCALE * 0.5)
        pygame.draw.circle(self.surf, (255, 0, 0), np.asarray(self.snake_head_position) * self.SCREEN_SCALE, self.SCREEN_SCALE * 0.5)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
