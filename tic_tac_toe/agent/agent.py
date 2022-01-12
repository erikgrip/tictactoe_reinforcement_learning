import numpy as np

class Agent:
    def __init__(self, strategy, model):
        self._strategy = strategy
        self._model = model


    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, s):
        self._strategy = s


    def get_action(self, state, valid_actions):
        qs = self._model.get_policy_qs(state)[0]
        aqs = list(enumerate(qs))  # Assuming 0 based action space
        # Pass only valid actions, with their q values (a, q)
        valid_qs = np.array([a_q for a_q in aqs if a_q[0] in valid_actions])
        action = self.strategy.select_action(valid_qs)
        return action
