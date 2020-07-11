import numpy as np
import random
from PIL import Image
import cv2
import os
import tensorflow as tf
from collections import deque
import time
from keras.models import  Sequential, load_model
from keras.callbacks import TensorBoard
from keras.layers import (Activation, Conv2D, Dense, Flatten, Dropout)
from keras.optimizers import Adam
from tqdm import tqdm
import math
from collections import namedtuple
import shelve

# Model path or False to create new models
AGENT_MODEL = 'models/2x256c_1vs1_vshuman___-0.70avg___-1.00min__1591217301.model'
# Strategy exploration settings
MAX_EPSILON = 0 # not a constant, going to be decayed
EPSILON_DECAY = 0.01
MIN_EPSILON = .0

# Environment settings
MODE = 'human'
EPISODES = 50
OPPONENT_MODEL = 'models/2x256c_1vs1_vshuman___-0.38avg___-1.00min__1591043963.model'
OPPONENT_RANDOM_RATE=0.5  # Apply when mode is 'model'. Between 0 and 1

# Replay memory settings
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100  # Wait until this size before sampling
MINIBATCH_SIZE = 64

# Learning settings
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 100  # Terminal states (end of episodes)

#  Model save and stats settings
MODEL_NAME = '2x256c_1vs1_vshuman'
MIN_REWARD = -200  # For model save
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

RGB_NORM = 255  # For scaling RGB images

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')
if not os.path.isdir('replay_history'):
    os.makedirs('replay_history')


# Own Tensorboard class - to override behaviour of writing a log file for
# every fit. We'd get more log files than what would be practical.
# Access with tensorboard --logdir=../../proj/tic_tac_toe/logs/ --host localhost --port 8088
class ModifiedTensorBoard(TensorBoard): 
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)



class TicTacToeEnv:    
    def __init__(self, mode='random', model=None, model_random_rate=0):
        self.game_size = 3
        self.action_space_size = self.game_size * self.game_size
        self.obs_space_values = (self.game_size, self.game_size, 3)
        self.mode = mode
        self.model = model
        self.model_random_rate = model_random_rate
        self.win_reward = 1
        self.draw_reward = 0
        self.loss_penalty = -1
        self.player_colors = {0: [0, 0, 0],
                              1: [255, 175, 0],
                              2: [0, 255, 0]}
        self.valid_actions = []
        self.game_history = []

    def reset(self):
        self.board = [[0 for i in range(self.game_size)] \
                      for i in range(self.game_size)]
        self.game_history = []
        self.valid_actions = list(range(self.game_size * self.game_size))
        self.agent_plays_first = random.choice([0, 1])
        self.done = False
        if not self.agent_plays_first:
            self.make_opponent_move(2)
        observation = np.array(self.get_image())
        return observation
    
    def get_game_history(self):
        return self.game_history

    def step(self, action): 

        agent_player = 1
        opponent_player = 2
        
        self.play_pawn(action, agent_player)
 
        reward = 0
        if self.win(self.board):
            reward = self.win_reward
            self.done = True
        elif self.draw(self.board):
            reward = self.draw_reward
            self.done = True

        if not self.done:
            # Make opponent move
            self.make_opponent_move(opponent_player)
            
            if self.win(self.board):
                reward = self.loss_penalty
                self.done = True
            elif self.draw(self.board):
                reward = self.draw_reward
                self.done = True

        new_observation = np.array(self.get_image())
        valid_actions = self.valid_actions[:] 

        return new_observation, reward, valid_actions, self.done
    
    def make_opponent_move(self, player):
        if self.mode == 'human':
            self.render()
            action = -1
            while action not in self.valid_actions:
                try:
                    action = int(input(f'Choose action {self.valid_actions}:'))
                except:
                    pass
        elif self.mode == 'model':
            if np.random.random() > self.model_random_rate:
                action = self.model.get_best_valid_action(
                    np.array(self.get_image()), self.valid_actions)
            else: 
                action = np.random.choice(self.valid_actions)
        elif self.mode == 'random':
            action = random.choice(self.valid_actions)
        else:
            raise ValueError
                
        self.play_pawn(action, player)
    
    def play_pawn(self, action, player):
        if self.is_valid_move(action):
            if action == 0:
                row, col = (0, 0)
            elif action == 1:
                row, col = (0, 1)
            elif action == 2:
                row, col = (0, 2)
            elif action == 3:
                row, col = (1, 0)
            elif action == 4:
                row, col = (1, 1)
            elif action == 5:
                row, col = (1, 2)
            elif action == 6:
                row, col = (2, 0)
            elif action == 7:
                row, col = (2, 1)
            elif action == 8:
                row, col = (2, 2)
       
        self.board[row][col] = player
        self.game_history.append((action, player))
        self.valid_actions.remove(action)
        
    def is_valid_move(self, action):
        if action in self.valid_actions:
            return True
        else:
            raise ValueError
            print('Invalid move')
            return False

    def render(self):
        img = self.get_image()
        # resizing so we can see our agent in all its glory.
        img = img.resize((300, 300), resample=Image.NEAREST)
        
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        # Map colors from color dict to board for a 3 layer representation
        board_array = np.array(
            [list(map(self.player_colors.get, x)) for x in iter(self.board)],
            dtype=np.uint8)
        # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        img = Image.fromarray(board_array, 'RGB') 
        return img
    
    
    def win(self, current_game):
        def all_same(l):
            if l.count(l[0]) == len(l) and l[0] != 0:
                return True
            else:
                return False
        # Horizontal
        for row in current_game:
            if all_same(row):
                player = row[0]
                print(f"Player {player} won the game horizontally!")
                return True
        # Vertical
        for col in range(len(current_game)):
            check = []
            for row in current_game:
                check.append(row[col])
            if all_same(check):
                player = check[0]
                print(f"Player {player} won the game vertically!")   
                return True
        # Diagonal
        diags = []
        for ix in range(len(current_game)):
            diags.append(current_game[ix][ix])
        if all_same(diags):
            player = diags[0]
            print(f"Player {player} won the game diagonally!")   
            return True
        diags = []
        for col, row in enumerate(reversed(range(len(current_game)))):
            diags.append(current_game[row][col])
        if all_same(diags):
            player = diags[0]
            print(f"Player {player} won the game diagonally!") 
            return True
        
        return False
    
    def draw(self, current_game):
        if any(0 in row for row in current_game):
            return False
        else:
            return True

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)
            
            
class DQN():
    def __init__(self, saved_model=None, input_shape=None, num_outputs=None):
        if saved_model == None:
            self.num_outputs = num_outputs
            self.model = self.create_model(input_shape)
        else:
            self.model = saved_model
            self.num_outputs = self.model.get_output_shape_at(-1)[1]
            
    def create_model(self, input_shape):
        model = Sequential()

        model.add(Conv2D(256, (3, 3),  input_shape=input_shape))
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
        
        model.add(Dense(self.num_outputs, activation='linear'))
        
        model.compile(optimizer=Adam(lr=0.001),
                      loss='mse', metrics=['accuracy'])
        return model 
    
    def get_qs(self, states):
        return self.model.predict(
            # Use [-3:] not to predict on batch size dimension
            np.array(states).reshape(-1,*states.shape[-3:]) / RGB_NORM)
    
    def get_best_valid_action(self, state, valid_actions):
        # An output may be invalid given state. Make sure move is legal.
        qs = self.get_qs(state)[0]  # [0] only list in list
        mask = np.zeros(self.num_outputs, dtype=int)  
        mask[valid_actions] = 1
        subset_idx = np.argmax(qs[mask == True])
        action = np.arange(qs.shape[0])[mask == True][subset_idx]
        return action
        
    
class DQNAgent:
    def __init__(self, strategy):
        self.strategy = strategy
        self.current_step = 0

    def get_current_epsilon(self):
        return self.strategy.get_exploration_rate(self.current_step)
    
    def select_action(self, state, model, valid_actions):
        threshold = self.strategy.get_exploration_rate(self.current_step)
        if np.random.random() > threshold:
            action = model.get_best_valid_action(state, valid_actions)
        else: 
            action = np.random.choice(valid_actions)
        
        self.current_step += 1
        return action

    
# We'll use experiences from replay memory to train the network.-Include next 
# valid action because else model won't know which future qvalues to discard
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'reward', 
     'next_state', 'next_valid_actions','is_terminal_state'))

# Create class to store the Experience objects
class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, experience):
            self.memory.append(experience)
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide(self, min_replay_memory_size):
        return len(self.memory) >= min_replay_memory_size
    

# Load opponent model if provided
if OPPONENT_MODEL:
    opponent_model = DQN(load_model(OPPONENT_MODEL))
else:
    opponent_model = None    
    
env = TicTacToeEnv(mode=MODE, 
                   model=opponent_model, 
                   model_random_rate=OPPONENT_RANDOM_RATE)
strategy = EpsilonGreedyStrategy(MAX_EPSILON, MIN_EPSILON, EPSILON_DECAY)
agent = DQNAgent(strategy)
memory = ReplayMemory(REPLAY_MEMORY_SIZE)

# Load saved games into replay memory
if MODE == 'human':
    saved_states = shelve.open("replay_history/replay_history") 
    for key in saved_states.keys():
        try:
            for experience in saved_states[key]:
                memory.push(experience)
        except EOFError:
            pass
        
    saved_states.close()
        
# Set up policy and target network
if AGENT_MODEL:
    policy_model = DQN(load_model(AGENT_MODEL))
    target_model = DQN(load_model(AGENT_MODEL))
else:
    policy_model = DQN(None, env.obs_space_values, env.action_space_size)
    target_model = DQN(None, env.obs_space_values, env.action_space_size)
    target_model.model.set_weights(policy_model.model.get_weights()) 


tensorboard = ModifiedTensorBoard(
    log_dir=f"logs/{MODEL_NAME}-{int(time.time())}") 

saved_games = []
ep_rewards = [] 
for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episode'):
    tensorboard.step = episode
    episode_reward = 0
    current_state = env.reset()
    if SHOW_PREVIEW:
        env.render()
    
    done = False
    while not done:
        action = agent.select_action(current_state, 
                                     policy_model,
                                     env.valid_actions)
        next_state, reward, next_valid_actions, done = env.step(action)
        memory.push(Experience(current_state, action, reward, 
                               next_state, next_valid_actions, done))
            
        epsilon = agent.get_current_epsilon()   
        if SHOW_PREVIEW:
            env.render()

        if memory.can_provide(MIN_REPLAY_MEMORY_SIZE):
            experiences = memory.sample(MINIBATCH_SIZE)
            # Extract states, actions, rewards and next_states into 
            #their own tensors from a given Experience batch
            current_states = np.array(
                [experience[0] for experience in experiences])
            current_qs_list = policy_model.get_qs(current_states)
            new_states = np.array(
                [experience[3] for experience in experiences])
            future_qs_list = target_model.get_qs(new_states)
         
            X = []
            y = []
        
            # The long tuple is the transistions in the minibatch
            for index, (
                    exp_state, exp_action, exp_reward, exp_next_state, 
                    exp_next_valid, exp_game_done) in enumerate(experiences):
            
                if not exp_game_done:
                    future_qs = future_qs_list[index]
                    # Make sure to only consider valid movas for next state
                    mask = np.zeros(future_qs.shape[0], dtype=int)
                    mask[exp_next_valid] = 1
                    max_future_q = np.max(future_qs[mask == True])    
                    new_q = exp_reward + DISCOUNT * max_future_q     
                else:
                    new_q = exp_reward
                    
                current_qs = current_qs_list[index]
                current_qs[exp_action] = new_q
                
                X.append(exp_state)
                y.append(current_qs)
            
            # Fit on all samples as one batch, log only on terminal state
            policy_model.model.fit(
                np.array(X) / RGB_NORM,
                np.array(y),
                batch_size=MINIBATCH_SIZE,
                verbose=0,
                shuffle=False,
                callbacks=[tensorboard] if exp_game_done else None)

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()
        
        episode_reward += reward
        current_state = next_state
        
    # Append episode reward to a list and log stats
    ep_rewards.append(episode_reward)
    
    saved_games.append(env.get_game_history())
        
    # Update target net to equal policy net
    if episode % UPDATE_TARGET_EVERY == 0:
        target_model.model.set_weights(policy_model.model.get_weights())
        
    # Update tensorboard and save model
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        latest_rewards = ep_rewards[-AGGREGATE_STATS_EVERY:]
        average_reward = sum(latest_rewards) / len(latest_rewards)
        min_reward = min(latest_rewards)
        max_reward = max(latest_rewards)
        pct_win = latest_rewards.count(env.win_reward) / len(latest_rewards)
        pct_draw = latest_rewards.count(env.draw_reward) / len(latest_rewards)
        pct_loss = latest_rewards.count(env.loss_penalty) / len(latest_rewards)
        tensorboard.update_stats(reward_avg=average_reward, 
                                 reward_min=min_reward, 
                                 reward_max=max_reward, 
                                 win_percent=pct_win,
                                 draw_percent=pct_draw,
                                 loss_percent=pct_loss,
                                 epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            model_name = f"models/{MODEL_NAME}_{average_reward:_>7.2f}avg_" \
                + f"{min_reward:_>7.2f}min__{int(time.time())}.model"
            policy_model.model.save(model_name)

if MODE == 'human':
    # Save replay memory to file
    replay_history = shelve.open("replay_history/replay_history")
    games_db = shelve.open("replay_history/saved_games") 
    replay_history[f"{MODEL_NAME}_{int(time.time())}"] = memory.memory
    games_db[f"{MODEL_NAME}_{int(time.time())}"] = saved_games
    replay_history.close()
    games_db.close()
            
 
    
