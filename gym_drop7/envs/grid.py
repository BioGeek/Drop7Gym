import numpy as np
import random
import cfg
from matplotlib import colors
import matplotlib.pyplot as plt
from itertools import groupby
import time


class Grid(object):
    last_frame_time = 0
    chain = {1: 7, 2: 39, 3: 109, 4: 224, 5: 391, 6: 617, 7: 907, 8: 1267, 9: 1701, 10: 2207}

    def __init__(self, stats, mode = "classic", demo=False, grid_size=7):
        self.grid_size = grid_size
        self.stats = stats

        self.grid = np.zeros((grid_size, grid_size), dtype=np.int)
        self.generate_init_grid(self.grid_size, mode)
        self.next_ball = generate_next_ball(self.grid_size)
        self.step_count = 0

    def generate_init_grid(self, _size, mode):
        """
        drop-7 starting grid with a few rules.

        Explode as needed (vertical)
        Explode as needed (horizontal)
        """
        
        if mode == "classic":
            for x in np.nditer(self.grid, op_flags=['readwrite']):
                # generate a U(0,1). If it is less than _fraction, then get a random integer from 1..7
                if random.random() <= random.uniform(cfg._FRACTION[0], cfg._FRACTION[1]):
                    x[...] = random.randint(1, _size)  # ellipsis will modify the right element
        
        elif mode == "sequence":
            self.grid[-1, :] = generate_fixed_row(cfg._SIZE, -2)
        
        # apply gravity to each column
        for col_num in range(self.grid.shape[1]):
            _, _, new = apply_gravity_to_column(self.grid[:, col_num])
            self.grid[:, col_num] = new

        # s = _Stats
        self.update_grid()
        # s.reset(cfg._outfile)

    def grid_as_array(self):
        return np.array(self.grid)

    def award_points(self, chain_level, explosions):
        points = 0
        if chain_level > 10:
            chain_level = 10
        points += self.chain[chain_level] * explosions
        return points
        # self.ptslist.append(self.chain[chain_level] * explosions)
        # self.explist.append(explosions)

    def update_grid(self):
        """
        Main function to update the grid

        """
        gravity_done, explosions_done = 0, 0
        chain_level = 0
        reward = 0
        while not (gravity_done and explosions_done):
            chain_level += 1
            explosions_done, reward_temp = self.apply_explosions_to_grid(chain_level)
            reward += reward_temp
            gravity_done = self.apply_gravity_to_grid()
            # print("In update grid", explosions_done, gravity_done)

        self.next_ball = generate_next_ball(self.grid_size)
        return reward

    def apply_explosions_to_grid(self, chain_level):
        original = self.grid.copy()  # need this for calculating points
        reward = 0
        # explosions = 0
        # for each row, calculate explosions (but don't execute them)
        # for each col, caluclate explosions (but don't execute them)
        row_mask, col_mask = grid_of_ones(cfg._SIZE), grid_of_ones(cfg._SIZE)
        for i in range(cfg._SIZE):
            _, _, row_mask[i, :] = inplace_explosions(self.grid[i, :])
            _, _, col_mask[:, i] = inplace_explosions(self.grid[:, i])

        # Executing all the explosions at once
        for i in range(cfg._SIZE):
            self.grid[i, :] = self.grid[i, :] * row_mask[i, :]
            self.grid[:, i] = self.grid[:, i] * col_mask[:, i]

        # Explosions is the NUMBER of BALLS that EXPLODE at a give grid configuration
        explosions = np.count_nonzero(original != self.grid)
        explosions_done = (explosions == 0)
        if chain_level >= 1:
            print("Chain Level:", chain_level, file=open(cfg._outfile, "a"))
            reward = self.award_points(chain_level, explosions)
        return explosions_done, reward

    def apply_gravity_to_grid(self):
        original = self.grid.copy()
        for i in range(cfg._SIZE):
            _, _, self.grid[:, i] = apply_gravity_to_column(self.grid[:, i])

        updated = self.grid.copy()
        return np.array_equal(updated, original)

    def row(self, rnum, _string):
        self.grid[cfg._SIZE - 1 - rnum, :] = list(_string)

    def show_grid(self):
        # Function used to render the game screen
        # Get the last rendered frame

        # make a color map of fixed colors
        cmap = colors.ListedColormap(
            ['darkgray', 'lightgray', 'white', 'green', 'yellow', 'orange', 'red', 'purple', 'lightblue', 'blue'])
        bounds = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]  # where to separate the colors
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        # im = ax.imshow(grid)
        for (j, i), label in np.ndenumerate(self.grid):
            if label != 0:
                ax.text(i, j, label, ha='center', va='center')

        plt.imshow(self.grid.reshape((cfg._SIZE,) * 2),
                   interpolation='none', cmap=cmap, norm=norm)

    def is_grid_full(self):
        nz = np.count_nonzero(self.grid)
        return nz == (cfg._SIZE * cfg._SIZE)

    def drop_ball_in_column(self, ball, col):
        """
        If valid column, find the first zero in the column and replace the value there.
        If column is full, return illegal flag
        If grid is full game_over
        """

        self.step_count += 1
        special_reward = 0
        if self.step_count % cfg._BALLS_TO_LEVELUP == 0:
            game_over = self.level_up()
            special_reward = 7000
            if game_over:
                return True, False, 0

        game_over = self.is_grid_full()
        gcol = self.grid[:, col]
        slot = np.where(gcol == 0)[0]
        if not slot.size:  # returned []
            need_another_col = True
        else:
            need_another_col = False

        if not game_over and not need_another_col:
            self.grid[slot[-1], col] = ball  # place in the last zero column, from the top

        if game_over:
            need_another_col = False

        return game_over, need_another_col, special_reward

    def level_up(self):
        """
        Add a row of balls to the bottom of the grid.
        If the top row has any ball, Game Over

        Note that when the balls are first surfaced, they are all -2. (meaning doubly hidden)
        The first explosion exposes them once to -1.
        Second neigboring explosion makes them the Original value.
        """
        # if top row has something, return grid and gameover
        if top_row_occupied(self.grid):
            return True  # game over

        original = self.grid.copy()
        for i in range(cfg._SIZE - 1):
            self.grid[i, :] = original[i + 1, :]  # move the row up

        self.grid[-1, :] = generate_random_row(cfg._SIZE)
        self.grid[-1, :] = generate_fixed_row(cfg._SIZE, -2)

        return False  # game is not over

    def print_game_over(self, s):
        print("GAME OVER")
        print(self.grid)
        print(np.count_nonzero(self.grid))
        print("DONE")
        print(s.ball_count, s.levelup_count, s.points)


def generate_next_ball(size):
    return random.randint(1, size)


def generate_random_row(_size):
    return np.random.randint(low=1, high=_size + 1, size=(1, _size))


def generate_fixed_row(_size, fixed_num):
    return np.asarray([fixed_num, ] * _size)


def top_row_occupied(grid_num):
    return np.count_nonzero(grid_num[0, :])


def blank_out(_num, vec):
    return [0 if x == _num else x for x in vec]


def mask(vec):
    return [x > 0 for x in vec]


def current_time_in_mill():
    return int(round(time.time() * 1000))


def set_max_fps(last_frame_time, _fps=1.0):
    current_milli_time = current_time_in_mill()
    remaining_sleep_time = 1. / _fps - (current_milli_time - last_frame_time) / 1000.
    if remaining_sleep_time > 0:
        time.sleep(remaining_sleep_time)
    return current_milli_time


def grid_of_zeros(size=cfg._SIZE):
    return np.zeros((size, size), dtype=np.int)


def grid_of_ones(size=cfg._SIZE):
    return np.ones((size, size), dtype=np.int)


def get_mask_lengths(_vec):
    """
    Outputs a tuple of rle lengths, 0's and 1's and their rle's
    """

    m = mask(_vec)
    b = range(len(m))
    ml = []
    for group in groupby(iter(b), lambda x: m[x]):  # use m[x] as the grouping key.
        ml.append((group[0], len(list(group[1]))))  # group[0] is 1 or 0. and group[1] is its rle

    return ml


def inplace_explosions(vec):

    exp_occurred = False

    original = [x for x in vec]  # manually creating a deepcopy
    updated_vec = [x for x in vec]  # manually creating a deepcopy

    ml = get_mask_lengths(updated_vec)  # number of contiguous non-zeros
    # print(ml)
    start, end = 0, 0
    for piece in ml:
        face_value, run_length = piece[0], piece[1]
        start = end
        end = start + run_length
        # print(vec[start:end])
        if face_value:  # True, nonzero elements exist
            seg = updated_vec[start:end]
            exploded_seg = blank_out(run_length, seg)
            if seg != exploded_seg:
                exp_occurred = True
                updated_vec[start:end] = exploded_seg[:]

    # this is a list of all the elements that remained unchanged. This is the !MASK of changes
    unchanged = [1 if i == j else 0 for i, j in zip(original, updated_vec)]

    # print("Exp occurred", exp_occurred)
    return exp_occurred, original, unchanged


def apply_gravity_to_column(column):
    """
    An entire column is adjusted for 'gravity.' All the zeros float to the top.
    All the other numbers come down as needed.
    """

    original = column[:]  # original
    updated = column[:]  # this can be changed
    flip_flag = 1
    safety_brkr = 0
    flip_occurred = False

    while flip_flag:
        a = updated[:]
        safety_brkr += 1
        flip_flag = 0  # off
        for index, (up, down) in enumerate(zip(a[:-1], a[1:])):
            if up and not down:
                updated[index], updated[index + 1] = 0, up
                # print("After ", index, "Column looks like:", column)
                flip_flag = 1  # at least one flip happened, so keep going
                flip_occurred = True
                if safety_brkr >= 100:
                    flip_flag = 0

    return flip_occurred, original, updated
