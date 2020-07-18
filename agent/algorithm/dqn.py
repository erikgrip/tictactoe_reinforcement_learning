from keras.models import  Sequential
from keras.layers import Activation, Conv2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
import numpy as np

class DQN():
    def __init__(self, memory=None, in_dim=None, out_dim=None, 
                 saved_model=None):
        self.memory=memory
        if saved_model == None:
            self.out_dim = out_dim
            self.policy_model = self.create_model(in_dim)
            self.target_model = self.create_model(in_dim)
            self.update_target_weights()
        else:
            self.policy_model = saved_model
            self.target_model = saved_model
            self.out_dim = self.policy_model.get_output_shape_at(-1)[1]
            
    def create_model(self, in_dim):
        model = Sequential()

        model.add(Conv2D(256, (3, 3),  input_shape=in_dim))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        
        model.add(Flatten())
        '''
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256))
        model.add(Dense(256))
        '''
        
        model.add(Dense(self.out_dim, activation='linear'))
        
        model.compile(optimizer=Adam(lr=0.001),
                      loss='mse', metrics=['accuracy'])
        return model

    def __get_qs(self, model, states):
        return model.predict(
            # Use [-3:] not to predict on batch size dimension
            np.array(states).reshape(-1,*states.shape[-3:]))
    
    def get_policy_qs(self, states):
        return self.__get_qs(self.policy_model, states)
    
    def get_target_qs(self, states):
        return self.__get_qs(self.target_model, states)
    
    def train(self, experiences, discount, game_done, callbacks=None):
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
                    new_q = reward + discount * max_future_q     
                else:
                    new_q = reward
                    
                current_qs = current_qs_list[index]
                current_qs[action] = new_q
                
                X.append(state)
                y.append(current_qs)
            
            print(len(experiences))
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
        
    
    


