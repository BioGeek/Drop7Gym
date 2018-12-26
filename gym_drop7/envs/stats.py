import cfg


class Stats(object):
    """
    Object to keep count of reward and general game statistics
    TODO: Implement reward function into class object
    """

    # chain points -> e.g. making a chain reaction of 2 results in 39 points * explosions of blocks


    def __init__(self):
        self.reset(cfg._outfile)

    def reset(self, _outfile):
        self.ball_count = 0
        self.levelup_count = 0
        self.points = 0
        self.nz = []
        self.ptslist, Stats.explist = [], []

        print("", file=open(_outfile, "w"))

    def ball_drop(self):
        self.ball_count += 1

    def incr_level_count(self):
        self.levelup_count += 1

    def update_nz(self, numnz):
        self.nz.append(numnz)

if __name__ == '__main__':
    game = Stats()
    print(game.__dict__)