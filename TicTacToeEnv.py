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
        self.board = \
            [[0 for i in range(self.game_size)] for i in range(self.game_size)]
        self.valid_actions = list(range(self.game_size * self.game_size))
        self.done = False
        observation = np.array(self.board)
        return observation

    def step(self, action): 
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


