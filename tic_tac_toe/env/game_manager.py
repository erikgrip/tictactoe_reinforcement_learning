from tic_tac_toe.env.tic_tac_toe_env import TicTacToe
from tic_tac_toe.agent.agent import Agent
import random
import numpy as np
from PIL import Image
import cv2

class TicTacToeGameManager():
    def __init__(self, strategy=None, saved_model=None):
        self.game = TicTacToe()
        self.agent_first_cmap = {0: 177, 1: 255, 2: 0}
        self.agent_last_cmap = {0: 177, 1: 0, 2: 255}


    def reset(self):
        self.game_history = []
        observation = self.game.reset()
        self.agent_plays_first = random.choice([0, 1])
        if not self.agent_plays_first:
            self.step(self.get_env_action(observation))
        observation = self.get_obs()
        return observation


    def step(self, action):
        _, reward, done = self.game.step(action)
        new_observation = self.get_obs()
        return new_observation, reward, done


    def print_board(self):
        self.game.print_board()


    def render(self):
        self.game.render()


    def get_env_action(self, state):
        return random.choice(self.valid_actions())


    def valid_actions(self):
        return self.game.valid_actions[:]


    def action_space_size(self):
        return self.game.action_space_size


    def obs_space_size(self):
        return self.game.obs_space_values


    def game_history(self):
        return self.game_history


    def win_reward(self):
        return self.game.win_reward


    def draw_reward(self):
        return self.game.draw_reward


    def loss_penalty(self):
        return self.game.loss_penalty


    def get_image(self):
        # Map colors from color dict to board, keeping agent color constant
        if self.agent_plays_first:
            cmap = self.agent_first_cmap
        else:
            cmap = self.agent_last_cmap
        board_array = np.array(
            [list(map(cmap.get, x)) for x in iter(self.game.board)],
            dtype=np.uint8)
        img = Image.fromarray(board_array, 'L')
        return img


    def get_obs(self):
        # Give shape (3, 3, 1) instead of (3, 3) for grayscale image
        obs = np.expand_dims(np.array(self.get_image()), axis=2)
        return obs
