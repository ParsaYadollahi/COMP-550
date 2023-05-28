import math
from collections import defaultdict


class Collocation:
    def __init__(self, word):
        self.word = word
        self.frequency = defaultdict(lambda: 0)

    def col(self):
        tot_freq = sum(self.frequency.values())
        max_freq = max(self.frequency.values())
        ret = abs(math.log((max_freq + 0.1) / tot_freq))
        return ret

    def sense(self):
        tot_max = max(self.frequency.values())

        for sense, local_max in self.frequency.items():
            if local_max == tot_max:
                return sense

        return None
