from keras.models import Sequential
from keras.layers import Activation, Conv2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
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

    def get_target_qs(self, states) -> np.array:
        return self._get_qs(self.target_model, states)

    def train(self, experiences, game_done, callbacks=None):
            batch_size = len(experiences)
            # Extract states, actions, rewards and next_states into
            #their own tensors from a given Experience batch
            current_states = np.array(
                [experience[0] for experience in experiences])
            current_qs_list = self.get_policy_qs(current_states)
            new_states = np.array(
                [experience[3] for experience in experiences])
            future_qs_list = self.get_target_qs(new_states)

            X = []
            y = []

            # The long tuple is the transistions in the minibatch
            for index, (
                    state, action, reward, next_state,
                    next_valid, terminal_state) in enumerate(experiences):

                if not terminal_state:
                    future_qs = future_qs_list[index]
                    # Only consider valid moves for next state
                    mask = np.zeros(future_qs.shape[0], dtype=int)
                    mask[next_valid] = 1
                    max_future_q = np.max(future_qs[mask == True])
                    new_q = reward + self.discount * max_future_q
                else:
                    new_q = reward

                current_qs = current_qs_list[index]
                current_qs[action] = new_q

                X.append(state)
                y.append(current_qs)

            # Fit on all samples as one batch, log only on terminal state
            self.policy_model.model.fit(
                np.array(X),
                np.array(y),
                batch_size=batch_size,
                verbose=0,
                shuffle=False,
                callbacks=callbacks if game_done else None)

    def update_target_weights(self):
        self.target_model.model.set_weights(
            self.policy_model.model.get_weights())
