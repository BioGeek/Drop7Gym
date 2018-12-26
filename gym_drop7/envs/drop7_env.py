import numpy as np
from .grid import Grid
from .stats import Stats
import gym
from gym import spaces


class Drop7Env(gym.Env):
    """
    Class Drop7 is the actual game.
    """

    def __init__(self, grid_size=7):
        self.grid_size = grid_size
        self.stats = Stats()
        self.grid = Grid(self.stats)
        self.reset()

        # openai gym setup
        # Agent can drop the next disc into columns 0 - grid_size
        self.action_space = spaces.Discrete(self.grid_size)

        # Agent sees a tuple:
        # 1. the next ball to be dropped
        # 2. the grid as a box
        self.observation_space = spaces.Tuple(
            spaces.Discrete(self.grid_size),
            spaces.Box(low=-2, high=self.grid_size, dtype=np.float32)
        )

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []

    def render(self, mode='human'):
        self.grid.show_grid()

    def step(self, action):
        pass

    def _update_state(self, action):
        """
        Input: action and states
        Output: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size - 1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self.state = out

    def _draw_state(self):
        im_size = (self.grid_size,) * 2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2] - 1:state[2] + 2] = 1  # draw basket
        return canvas

    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size - 1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        if self.state[0, 0] == self.grid_size - 1:
            return True
        else:
            return False

    def observe(self):
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        n = np.random.randint(0, self.grid_size - 1, size=1)
        m = np.random.randint(1, self.grid_size - 2, size=1)
        self.state = np.asarray([0, n, m])[np.newaxis]

    def get_state(self):
        return self.state

if __name__ == '__main__':
    game = Drop7Env()
    print(game.grid)