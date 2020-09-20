from tic_tac_toe.agent.strategy.epsilon_greedy import EpsilonGreedyStrategy

class MaxStrategy(EpsilonGreedyStrategy):
    def __init__(self):
        super().__init__(start=0, end=0, decay=0)