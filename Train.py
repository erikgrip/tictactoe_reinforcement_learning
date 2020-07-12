from env.game_manager import TicTacToeGameManager
from agent.agent import Agent
from agent.algorithm.dqn import DQN
from util.utils import (
    sequential_model_from_spec, strategy_from_spec, memory_from_spec)
from util.tensorboard_mod import ModifiedTensorBoard
import os
import time
from tqdm import tqdm
from collections import namedtuple
import shelve
from keras.models import load_model
import argparse

default_spec = {
    'environment': {
        'name': 'TicTacToeGameManager',
        'mode': 'random'
    },
    'strategy': {
        'type': 'EpsilonGreedyStrategy',
        'max': 1,
        'min': 0,
        'decay': 0.001
    },
    'net': {
        'name': 'my_model',
        'layers':  [
            {'class_name': 'Flatten',
             'config': 
                 {'input_shape': (3,3,1),  # Make dynamic by ref to env.obs_size
                  }
            },
            {'class_name': 'Dense',
             'config': 
                 {'units': 64,
                  'activation': 'relu'
                  }
            },
            {'class_name': 'Dense',
             'config': 
                 {'units': 4,
                  'activation': 'softmax'
                  }
            }
        ],
        'lr':  0.02,
        'loss': 'mse',
        'keras_version': '2.3.1'
    },
    'replay_memory': {
        'type': 'StandardReplayMemory',
        'size': 50_000
    },
    'algorithm': {
        'type': 'DQN',
        'target_net_update_freq': 500,
        'discount': 0.99
    },
    'train': {
        'num_episodes': 10_000,
        'minibatch_size': 64,
        'min_memory': 500
    }
}
    

# Model path or False to create new models
AGENT_MODEL = False

# Environment settings
OPPONENT_MODEL = False

#  Model save and stats settings
AGGREGATE_STATS_EVERY = 50  # episodes

def main(spec):
    
    MODE = spec['environment']['mode']
    MODEL_NAME = spec['net']['name']
    EPISODES = spec['train']['num_episodes']
    MIN_MEMORY_TO_TRAIN = spec['train']['min_memory']
    MINIBATCH_SIZE = spec['train']['minibatch_size']
    DISCOUNT = spec['algorithm']['discount']
    UPDATE_TARGET_EVERY = spec['algorithm']['target_net_update_freq']
    
    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')
    if not os.path.isdir('replay_history'):
        os.makedirs('replay_history')
        
    tensorboard = ModifiedTensorBoard(
        log_dir=f"logs/{MODEL_NAME}-{int(time.time())}") 
    
    # We'll use experiences from replay memory to train the network.Include next 
    # valid action because else model won't know which future qvalues to discard
    Experience = namedtuple(
        'Experience',
        ('state', 'action', 'reward', 
         'next_state', 'next_valid_actions','is_terminal_state'))
                    
    
    env = TicTacToeGameManager(mode=MODE)
    strategy = strategy_from_spec(spec['strategy'])
    memory = memory_from_spec(spec['replay_memory'])      
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
            if memory.can_provide(MIN_MEMORY_TO_TRAIN):
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
        

if __name__ == '__main__':
    
    #parser = argparse.ArgumentParser()
    #parser.add_argument('spec')
    #args = parser.parse_args()
    
    args = default_spec  # Just for testing, remove later
    
    main(args)

    
    
    
    

            
 
    

