from agent.strategy.base_strategy import BaseStrategy
import numpy as np

class EpsilonGreedyStrategy(BaseStrategy):
    def __init__(self, start, end, decay):
        super().__init__(start, end, decay)

    def select_action(self, aq_pairs):
        threshold = self.get_decayed_rate()
        try:
            if (threshold == 0) or (np.random.random() > threshold):
                # Get action corresponding to maximum q
                action = aq_pairs[np.argmax(aq_pairs[:, 1])][0]
            else:
                action = np.random.choice(aq_pairs[:, 0])
        except:
            raise ValueError(f"{aq_pairs} not valid action/q-value array")

        self.current_step += 1
        return int(action)
