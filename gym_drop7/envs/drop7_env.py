import numpy as np
from .grid import Grid
from .stats import Stats
import gym
from gym import spaces


class Drop7Env(gym.Env):
    """
    Class Drop7 is the actual game.
    """

    def __init__(self, mode="classic", grid_size=7):
        self.grid_size = grid_size
        self.mode = mode
        self.stats = Stats()
        self.grid = Grid(self.stats, mode)

        # openai gym setup
        # Agent can drop the next disc into columns 0 - grid_size
        self.action_space = spaces.Discrete(self.grid_size)

        # Agent sees a tuple:
        # 1. the next ball to be dropped
        # 2. the grid as a box
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.grid_size),
            spaces.Box(low=-2, high=self.grid_size, shape=(self.grid_size, self.grid_size), dtype=np.float32)
        ))

        # Store what the agent tried
        self.curr_step = -1
        self.curr_episode = -1
        self.action_episode_memory = []

    def render(self, mode='human'):
        self.grid.show_grid()

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        # special_reward is used for level_up bonus
        game_over, need_another_col, special_reward = self.grid.drop_ball_in_column(self.grid.next_ball, action)
        
        ob = self.get_state()
        
        if game_over:
            # raise RuntimeError("Episode is done")
            reward = 0.0
        else:
            self.curr_step += 1
            reward = self.grid.update_grid() + special_reward

        return ob, reward, game_over, {}

    def get_state(self):
        next_ball = self.grid.next_ball
        return next_ball, self.grid.grid_as_array()

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

    def reset(self):
        self.curr_step = -1
        self.curr_episode += 1

        self.stats = Stats()
        self.grid = Grid(self.stats, self.mode)

        return self.get_state()
        # n = np.random.randint(0, self.grid_size - 1, size=1)
        # m = np.random.randint(1, self.grid_size - 2, size=1)
        # self.state = np.asarray([0, n, m])[np.newaxis]

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

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over


if __name__ == '__main__':
    game = Drop7Env()
    print(game.grid)