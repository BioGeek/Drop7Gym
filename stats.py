import cfg

class Stats(object):
    """
    Object to keep count of reward and general game statistics
    TODO: Implement reward function into class object
    """

    # chain points -> e.g. making a chain reaction of 2 results in 39 points * explosions of blocks
    chain = {1: 7, 2: 39, 3: 109, 4: 224, 5: 391, 6: 617, 7: 907, 8: 1267, 9: 1701, 10: 2207}

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

    def award_points(self, chain_level, explosions):
        self.points += self.chain[chain_level] * explosions
        self.ptslist.append(self.chain[chain_level] * explosions)
        self.explist.append(explosions)

    def update_nz(self, numnz):
        self.nz.append(numnz)

if __name__ == '__main__':
    game = Stats()
    print(game.__dict__)