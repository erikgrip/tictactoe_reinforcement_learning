from agent.strategy.base import BaseStratregy
import numpy as np

class Boltzmann(BaseStratregy):
    def __init__(self, start, end, decay):
        super().__init__(start, end, decay)
         
    def action_probs(self, aq_pairs):
        # As defined by Graessner anf Keng p.86. 
        tau = self.get_decayed_rate()
        tau = np.max([tau, 0.001])  # Tau=0 will lead to division by zero
        ps = (np.exp(aq_pairs[:, 1])/tau) / (np.exp(aq_pairs[:, 1]).sum()/tau)
        return ps
    
    def select_action(self, aq_pairs):
        ps = self.action_probs(aq_pairs)
        sampled_index = np.argmax(np.random.multinomial(n=1, pvals=ps))
        action = int(aq_pairs[sampled_index][0])
        self.current_step += 1
        return action


