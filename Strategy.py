''' Strategies are balancing exploitation and exploration. '''
import math
import numpy as np

class BaseStratregy():
    def __init__(self, start, end, decay):
        # Basic input validation
        if (start < 0) | (end < 0) | (decay < 0) :
            raise ValueError("Only positive arguments accepted")
        elif start < end:
            raise ValueError("End must not be greater than start")

        self.start = start
        self.end = end
        self.decay = decay
        self.current_step = 0 
        
    def get_decayed_rate(self):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * self.current_step * self.decay)
        
        
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
    
    
class MaxStrategy(EpsilonGreedyStrategy):
    def __init__(self, start, end, decay):
        super().__init__(start, end, decay)
        
        
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
        
        

    


