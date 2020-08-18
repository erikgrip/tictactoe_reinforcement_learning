from tic_tac_toe.agent.strategy.base import BaseStratregy
import numpy as np

class EpsilonGreedyStrategy(BaseStratregy):
    def __init__(self, start, end, decay):
        super().__init__(start, end, decay)

    def select_action(self, aq_pairs):
        threshold = self.get_decayed_rate()
        if np.random.random() > threshold:
            # Get action corresponding to maximum q
            action = aq_pairs[np.argmax(aq_pairs[:, 1])][0]
        else: 
            action = np.random.choice(aq_pairs[:, 0])
        
        self.current_step += 1
        return int(action)