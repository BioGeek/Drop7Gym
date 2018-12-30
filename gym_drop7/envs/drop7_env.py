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

        if game_over:
            reward = 0.0
            return (self.grid.next_ball, self.grid.grid_of_zeros(self.grid_size)), reward, game_over, {}

        else:
            self.curr_step += 1
            reward = self.grid.update_grid() + special_reward
            ob = self.get_state()

        return ob, reward, game_over, {}

    def get_state(self):
        next_ball = self.grid.next_ball
        return next_ball, self.grid.grid_as_array()

    def reset(self):
        self.curr_step = -1
        self.curr_episode += 1

        self.stats = Stats()
        self.grid = Grid(self.stats, self.mode)

        return self.get_state()

if __name__ == '__main__':
    game = Drop7Env()
    print(game.grid)
