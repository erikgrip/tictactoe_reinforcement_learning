from GameManager import TicTacToeGameManager
from RLAgent import Agent
from Strategy import EpsilonGreedyStrategy, Boltzmann
from ReplayMemory import StandardReplayMemory
from Algorithm import DQN
from TensorBoardMod import ModifiedTensorBoard
import os
import time
from tqdm import tqdm
from collections import namedtuple
import shelve
from keras.models import load_model

# Model path or False to create new models
AGENT_MODEL = False

# Strategy exploration settings
MAX_EXPLORE = 1 # not a constant, going to be decayed
EXPLORE_DECAY = 0.001
MIN_EXPLORE = .05

# Environment settings
MODE = 'random'
EPISODES = 10_000
OPPONENT_MODEL = False
OPPONENT_RANDOM_RATE=0.5  # Apply when mode is 'model'. Between 0 and 1

# Replay memory settings
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 128  # Wait until this size before sampling
MINIBATCH_SIZE = 128

# Learning settings
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 500  # Terminal states (end of episodes)

#  Model save and stats settings
MODEL_NAME = '2x256c_EpsG_vsrandom'
AGGREGATE_STATS_EVERY = 50  # episodes

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')
if not os.path.isdir('replay_history'):
    os.makedirs('replay_history')
    
tensorboard = ModifiedTensorBoard(
    log_dir=f"logs/{MODEL_NAME}-{int(time.time())}") 

# We'll use experiences from replay memory to train the network.-Include next 
# valid action because else model won't know which future qvalues to discard
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'reward', 
     'next_state', 'next_valid_actions','is_terminal_state'))
                

env = TicTacToeGameManager(mode=MODE)
strategy = EpsilonGreedyStrategy(MAX_EXPLORE, MIN_EXPLORE, EXPLORE_DECAY)
memory = StandardReplayMemory(REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE)        
# Set up policy and target network
if AGENT_MODEL:
    model = DQN(memory, saved_model=load_model(AGENT_MODEL))
else:
    model = DQN(memory=memory, 
                in_dim=env.obs_space_size(), 
                out_dim=env.action_space_size())
agent = Agent(strategy, model)

'''             
if MODE == 'human':
    # Load saved games into replay memory
    saved_states = shelve.open("replay_history/replay_history") 
    for key in saved_states.keys():
        try:
            for experience in saved_states[key]:
                agent.model.memory.push(experience)
        except EOFError:
            pass 
    saved_states.close()
'''
    
# Loop over episodes
saved_games = []
ep_rewards = [] 
for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episode'):
    tensorboard.step = episode
    episode_reward = 0
    
    current_state = env.reset()
    
    done = False
    while not done:
        # Select and perform action
        action = agent.get_action(current_state, env.valid_actions())
        next_state, reward, done = env.step(action)
        
        # Make opponent move
        if not done:
            env_action = env.get_env_action(next_state)
            next_state, reward, done = env.step(env_action)
            reward = -reward  # Assuming symmetric rewards
            
        next_valid_actions = env.valid_actions()
        
        # Store experience in replay memory
        model.memory.push(Experience(current_state/255, action, reward, 
                               next_state/255, next_valid_actions, done))
        
        
        # Train model
        if memory.can_provide():
            model.train(batch_size=MINIBATCH_SIZE,
                        discount=DISCOUNT, 
                        game_done=done,
                        callbacks=[tensorboard])
        
        # Get epsilon and update rewards and state
        episode_reward += reward
        current_state = next_state
        explore_param = agent.strategy.get_decayed_rate() 
        
    # Append episode reward to a list and log stats
    ep_rewards.append(episode_reward)
    
    # Keep game if played against human
    if MODE == 'human':
        saved_games.append(env.game_history)
        
    # Update target net to equal policy net
    if episode % UPDATE_TARGET_EVERY == 0:
        model.update_target_weights()
        
    # Update tensorboard and save model
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        latest_rews = ep_rewards[-AGGREGATE_STATS_EVERY:]
        average_reward = sum(latest_rews) / len(latest_rews)
        min_reward = min(latest_rews)
        max_reward = max(latest_rews)
        pct_win = latest_rews.count(env.win_reward()) / len(latest_rews)
        pct_draw = latest_rews.count(env.draw_reward()) / len(latest_rews)
        pct_loss = latest_rews.count(env.loss_penalty()) / len(latest_rews)
        tensorboard.update_stats(reward_avg=average_reward, 
                                 reward_min=min_reward, 
                                 reward_max=max_reward, 
                                 win_percent=pct_win,
                                 draw_percent=pct_draw,
                                 loss_percent=pct_loss,
                                 exploration_parameter=explore_param)

# Save model
model_name = f"models/{MODEL_NAME}_{average_reward:_>7.2f}avg_" \
    + f"{min_reward:_>7.2f}min_{int(time.time())}.model"
model.policy_model.model.save(model_name)

# Keep games if player manually
if MODE == 'human':
    # Save replay memory to file
    replay_history = shelve.open("replay_history/replay_history")
    games_db = shelve.open("replay_history/saved_games") 
    replay_history[f"{MODEL_NAME}_{int(time.time())}"] = memory.memory
    games_db[f"{MODEL_NAME}_{int(time.time())}"] = saved_games
    replay_history.close()
    games_db.close()
            
 
    

