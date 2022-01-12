from agent.strategy.base_strategy import BaseStrategy
import numpy as np

class Boltzmann(BaseStrategy):
    def __init__(self, start, end, decay):
        super().__init__(start, end, decay)

    def _action_probs(self, aq_pairs):
        # As defined by Graessner anf Keng p.86.
        tau = self.get_decayed_rate()
        tau = np.max([tau, 0.001])  # Tau=0 will lead to division by zero
        try:
            qs = np.array(aq_pairs[:, 1])
            # Normalize to avoid overflow. The output probability is
            # insensitive to shifts in values of qs
            qs = qs - qs.max()
        except:
            raise ValueError(f"Unable to find action for max Q in {aq_pairs}")
        ps = (np.exp(qs/tau)) / (np.exp(qs/tau).sum())
        return ps

    def select_action(self, aq_pairs):
        ps = self._action_probs(aq_pairs)
        sampled_index = np.argmax(np.random.multinomial(n=1, pvals=ps))
        action = int(aq_pairs[sampled_index][0])
        self.current_step += 1
        return action
