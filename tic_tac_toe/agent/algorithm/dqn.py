import numpy as np

class DQN():
    def __init__(self, policy_model, target_model, discount):
        self.policy_model = policy_model
        self.target_model = target_model
        self.update_target_weights()
        self.discount = discount


    def _get_qs(self, model, states) -> np.array:
        return model.predict(
            # Use [-3:] not to predict on batch size dimension
            np.array(states).reshape(-1,*states.shape[-3:]))


    def get_policy_qs(self, states) -> np.array:
        return self._get_qs(self.policy_model, states)


    def _valid_max_q(self, qs, valid_moves) -> float:
        mask = np.zeros(qs.shape[0], dtype=int)
        mask[valid_moves] = 1
        return np.max(qs[mask == True])


    def _prep_training_input(self, experiences):
        current_states = np.array([exp[0] for exp in experiences])
        future_states = np.array([exp[3] for exp in experiences])
        current_qs_list = self._get_qs(self.policy_model, current_states)
        future_qs_list = self._get_qs(self.target_model, future_states)

        X = []
        y = []
        for index, (state, action, reward, _, next_valid_moves,
                    is_terminal_state) in enumerate(experiences):

            if not is_terminal_state:
                future_qs = future_qs_list[index]
                # Only consider valid moves for next state
                max_future_q = self._valid_max_q(future_qs, next_valid_moves)
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(state)
            y.append(current_qs)
        return X, y


    def train(self, experiences, game_done, callbacks=None):
        X, y = self._prep_training_input(experiences)

        # Fit on all samples as one batch, log only on terminal state
        self.policy_model.model.fit(
            np.array(X),
            np.array(y),
            batch_size=len(experiences),
            verbose=0,
            shuffle=False,
            callbacks=callbacks if game_done else None)


    def update_target_weights(self):
        self.target_model.model.set_weights(
            self.policy_model.model.get_weights())
