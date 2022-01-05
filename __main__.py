from tic_tac_toe.env.game_manager import TicTacToeGameManager
from tic_tac_toe.agent.agent import Agent
from tic_tac_toe.agent.algorithm.dqn import DQN
from tic_tac_toe.agent.strategy.max_strategy import MaxStrategy
from tic_tac_toe.util.utils import (
    create_dir_if_not_exist,
    sequential_model_from_spec,
    strategy_from_spec,
    memory_from_spec,
    param_search_df_from_spec,
    df_row_to_spec,
    sequential_model_from_spec)
from tic_tac_toe.util.tensorboard_mod import ModifiedTensorBoard
import os
from pathlib import Path
import time
from tqdm import tqdm
from collections import namedtuple
import argparse
import json


# Environment settings
OPPONENT_MODEL = False

#  Model save and stats settings
AGGREGATE_STATS_EVERY = 50  # episodes

def main(input_spec):

    param_df = param_search_df_from_spec(input_spec)
    param_df['eval_avg'] = None  # For keeping evaluation run result
    PARAM_DF_EVAL_WINDOW = 1_000  # Evaluate on last 2k episodes i train run

    search_timestamp = int(time.time())

    # Loop over sample of parameter combinations from input spec
    for df_index, df_row in param_df.iterrows():

        spec = df_row_to_spec(df_row.drop('eval_avg'))
        EPISODES = spec['run']['num_episodes']
        MIN_MEMORY_TO_TRAIN = spec['replay_memory']['min_memory']
        MINIBATCH_SIZE = spec['replay_memory']['minibatch_size']
        UPDATE_TARGET_EVERY = spec['algorithm']['target_net_update_freq']
        MODEL_NAME = spec['net']['name']
        NET_SPEC = spec['net']
        DISCOUNT = spec['algorithm']['discount']

        tensorboard = ModifiedTensorBoard(
            log_dir=log_dir+f"{MODEL_NAME}-{run_ts}")

        # Include next_valid_action because else model won't know
        # which future q values to discard
        Experience = namedtuple(
            'Experience',
            ('state', 'action', 'reward',
             'next_state', 'next_valid_actions','is_terminal_state'))


        env = TicTacToeGameManager()
        strategy = strategy_from_spec(spec['strategy'])
        memory = memory_from_spec(spec['replay_memory'])
        model = DQN(policy_model=sequential_model_from_spec(NET_SPEC),
                    target_model=sequential_model_from_spec(NET_SPEC),
                    discount=DISCOUNT)
        agent = Agent(strategy, model)

        # Loop over training episodes
        ep_rewards = []
        for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episode'):
            tensorboard.step = episode

            episode_reward = 0
            current_state = env.reset()

            done = False
            while not done:
                action, next_state, reward, done, next_valid_actions = (
                    _play_turns(agent, env, current_state))

                # Store experience in replay memory
                memory.push(
                    Experience(current_state/255, action, reward,
                               next_state/255, next_valid_actions, done))

                # Train model
                if memory.can_provide(MIN_MEMORY_TO_TRAIN):
                    minibatch = memory.sample(MINIBATCH_SIZE)
                    model.train(minibatch,
                                game_done=done,
                                callbacks=[tensorboard])

                # Get epsilon and update rewards and state
                episode_reward += reward
                current_state = next_state

            # Append episode reward to a list and log stats
            ep_rewards.append(episode_reward)

            # Keep game if played against human
            if ENV_MODE == 'human':
                saved_games.append(env.game_history)

            # Update target net to equal policy net
            if episode % UPDATE_TARGET_EVERY == 0:
                model.update_target_weights()

            # Update tensorboard
            if not episode % NUM_EPISODES_TO_AGG_STATS or episode == 1:
                explore_param = agent.strategy.get_decayed_rate()
                new_rews = ep_rewards[-NUM_EPISODES_TO_AGG_STATS:]
                average_reward = sum(new_rews) / len(new_rews)
                min_reward = min(new_rews)
                max_reward = max(new_rews)
                pct_win = new_rews.count(env.win_reward()) / len(new_rews)
                pct_draw = new_rews.count(env.draw_reward()) / len(new_rews)
                pct_loss = new_rews.count(env.loss_penalty()) / len(new_rews)
                tensorboard.update_stats(reward_avg=average_reward,
                                         reward_min=min_reward,
                                         reward_max=max_reward,
                                         win_percent=pct_win,
                                         draw_percent=pct_draw,
                                         loss_percent=pct_loss,
                                         exploration_parameter=explore_param)

        # Save model
        model_file_name = f"{MODEL_NAME}-{run_ts}.model"
        model.policy_model.model.save(model_file_name)

        # Save spec
        spec_file_name = f"{MODEL_NAME}-{run_ts}-spec.json"
        with open(spec_file_name, 'w') as json_file:
            json.dump(spec, json_file)

        # Loop over evaluation episodes with no exploration
        agent = Agent(MaxStrategy(), model)
        for episode in range(1, PARAM_DF_EVAL_EPISODES+1):
                episode_reward = 0
                current_state = env.reset()

                done = False
                while not done:
                    action, next_state, reward, done, next_valid_actions = (
                        _play_turns(agent, env, current_state))

                    # Get epsilon and update rewards and state
                    episode_reward += reward
                    current_state = next_state

                # Append episode reward to a list and log stats
                ep_rewards.append(episode_reward)

        eval_reward_avg = sum(ep_rewards) / len(ep_rewards)
        param_df.loc[df_index, 'eval_avg'] = eval_reward_avg

    # Save df with the performance metric column eval_avg
    param_file_name = f"{MODEL_NAME}-{run_ts}.csv"
    param_df.to_csv(param_dir+param_file_name, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec')
    args = parser.parse_args()

    with open(args.spec, 'r') as json_file:
        spec = json.load(json_file)
    main(spec)
