from epsilon_greedy import EpsilonGreedyStrategy

class MaxStrategy(EpsilonGreedyStrategy):
    def __init__(self, start=0, end=0, decay=0):
        super().__init__(start, end, decay)


