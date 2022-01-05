import numpy as np

class TicTacToe:
    def __init__(self):
        self.game_size = 3
        self.action_space_size = self.game_size * self.game_size
        self.obs_space_values = (self.game_size, self.game_size, 1)
        self.win_reward = 1
        self.draw_reward = 0
        self.loss_penalty = -1
        self.reset()

    def reset(self):
        '''Reset game and return fresh board with no markers on it'''
        self.board = \
            [[0 for i in range(self.game_size)] for i in range(self.game_size)]
        self.valid_actions = list(range(self.game_size * self.game_size))
        self.done = False
        observation = np.array(self.board)
        return observation


    def step(self, action):
        """Take a move, square 0-8, and place a marker on the board."""
        if len(self.valid_actions) % 2 == 0:
            player = 2
        else:
            player = 1

        self.play_pawn(action, player)

        reward = 0
        if self.win(self.board):
            reward = self.win_reward
            self.done = True
        elif self.draw(self.board):
            reward = self.draw_reward
            self.done = True

        new_observation = np.array(self.board)
        return new_observation, reward, self.done


    def play_pawn(self, action, player):
        if action in self.valid_actions:
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
        else:
            raise ValueError

        self.board[row][col] = player
        self.valid_actions.remove(action)


    def print_board(self):
        print("---------")
        print(self.board[0])
        print(self.board[1])
        print(self.board[2])
        print("---------")


    def win(self, current_game):
        def all_elements_equal(a_list):
            if a_list.count(a_list[0]) == len(a_list) and a_list[0] != 0:
                return True
            else:
                return False

        def check_for_winner(a_list):
            if all_elements_equal(a_list):
                player = a_list[0]
                return True
            else:
                return False

        # Check horizontal win
        for row in current_game:
            if check_for_winner(row):
                return True
        # Check vertical win
        for col in range(self.game_size):
            col_markers = []
            for row in current_game:
                col_markers.append(row[col])
            if check_for_winner(col_markers):
                return True
        # Check diagonal win
        diags = []
        for ix in range(self.game_size):
            diags.append(current_game[ix][ix])
        if check_for_winner(diags):
            return True
        diags = []
        for col, row in enumerate(reversed(range(self.game_size))):
            diags.append(current_game[row][col])
        if check_for_winner(diags):
            return True

        return False


    def draw(self, current_game):
        if any(0 in row for row in current_game):
            return False
        else:
            return True
