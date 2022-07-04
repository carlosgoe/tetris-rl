import numpy as np
import random
from time import sleep
from IPython.display import clear_output


class Tetris:
    
    def __init__(self, gui=False):
        # All valid combinations of position and rotation (action = 4 * position + rotation)
        self.actions = list(range(34)) + [35, 37]
        # Further necessary attributes
        self.gui = gui
        self.bag = list(range(1, 8))
        self.positions = dict([(n, np.empty((0, 2), dtype=np.uint8)) for n in range(1, 8)])
        self.score = 0
        self.lines = 0
        # Select random tetrimino and show environment (if gui is enabled)
        self.new_tetrimino()
        self.show()
        
    def new_tetrimino(self):
        if self.bag == []:
            self.bag = list(range(1, 8))
        self.tetrimino = random.choice(self.bag)
        self.bag.remove(self.tetrimino)
        
    def get_tetrimino(self, rotation=0):
        # I (2 rotation states)
        if self.tetrimino == 1:
            return np.rot90(np.ones((1, 4)), rotation)
        # O (1 rotation state)
        if self.tetrimino == 2:
            return np.full((2, 2), 2)
        arr = np.zeros((2, 3))
        # J (4 rotation states)
        if self.tetrimino == 3:
            arr[0, 0] = 3
            arr[1] = 3
        # L (4 rotation states)
        if self.tetrimino == 4:
            arr[0, 2] = 4
            arr[1] = 4
        # S (2 rotation states)
        if self.tetrimino == 5:
            arr[[0, 0, 1, 1], [1, 2, 0, 1]] = 5
        # T (4 rotation states)
        if self.tetrimino == 6:
            arr[1] = 6
            arr[0, 1] = 6
        # Z (2 rotation states)
        if self.tetrimino == 7:
            arr[[0, 0, 1, 1], [0, 1, 1, 2]] = 7
        return np.rot90(arr, 4 - rotation)
    
    def get_board(self):
        board = np.zeros((20, 10), dtype=np.uint8)
        for k, v in self.positions.items():
            board[v[:, 0], v[:, 1]] = k
        return board

    def density(self):
        # Get positions of all filled cells
        all_pos = np.concatenate(list(self.positions.values()))
        # Return 1 if no cells are filled
        if all_pos.shape[0] == 0:
            return 1
        # Get maximum column height
        height = 20 - np.min(all_pos[:, 0])
        # Return proportion of filled cells in rectangle of size height x 10
        return all_pos.shape[0] / (height * 10)

    def invalid(self):
        # Return valid action indices of tetrimino
        if self.tetrimino == 1:
            return [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 28, 30, 31, 32, 34]
        if self.tetrimino == 2:
            return [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35]
        if self.tetrimino in (3, 4, 6):
            return [32, 35]
        if self.tetrimino in (5, 7):
            return [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 32, 34, 35]

    def obs(self):
        inp = np.zeros((20, 10))
        all_pos = np.concatenate(list(self.positions.values()))
        inp[all_pos[:, 0], all_pos[:, 1]] = self.tetrimino / 7
        return inp.flatten()
    
    def show(self):
        if self.gui:
            replacements = {'0': '▢', '1': '▣', '2': '▤', '3': '▥', '4': '▦', '5': '▧', '6': '▨', '7': '▩', '[': '', ']': ''}
            padding = '\n' if self.tetrimino == 1 else ''
            output = '{0} {1}\n\n {2}'.format(padding, self.get_tetrimino().astype(np.uint8), self.get_board())
            for key, val in replacements.items():
                output = output.replace(key, val)
            sleep(1)
            clear_output(wait=True)
            print(output + '\n\nLevel: {0}\nScore: {1}\nLines: {2}'.format(self.lines // 10, self.score, self.lines))
    
    def step(self, action_id):
        # Rotation of tetrimino (0 - 3)
        rot = self.actions[action_id] % 4
        # Current tetrimino as array
        tetr = self.get_tetrimino(rot)
        # Position of tetrimino = its top right cell
        pos = min(self.actions[action_id] // 4, 10 - tetr.shape[1])
        # Indices of board columns to be updated after dropping the tetrimino
        width_range = pos + np.arange(tetr.shape[1])
        # Free space of each board column in width_range to top
        free_top = np.argmax(np.concatenate([self.get_board()[:, width_range] > 0, np.ones((1, tetr.shape[1]))]), axis=0)
        # Free space of each tetrimino column to bottom
        free_bottom = np.argmax(np.flip(tetr > 0, 0), axis=0)
        # Minimum free space between tetrino and tetrinos in board
        min_space = np.min(free_top + free_bottom)
        # Game over if height of tetrimino exceeds min_space
        if min_space < tetr.shape[0]:
            return np.full(200, self.tetrimino / 7), 0, True, []
        # Calculate position of tetrimino cells in board
        new_pos = np.argwhere(tetr > 0)
        new_pos[:, 0] += min_space - tetr.shape[0]
        new_pos[:, 1] += pos
        # Update positions
        self.positions[self.tetrimino] = np.concatenate([self.positions.get(self.tetrimino), new_pos])
        # Get number of filled cells per row
        filled_per_row = np.sum(self.get_board() > 0, axis=1)
        cleared = 0
        while np.max(filled_per_row) == 10:
            # Get index of first full row from bottom
            i_row = np.argmax(filled_per_row)
            for k, v in self.positions.items():
                # Remove all positions with that index
                v = np.delete(v, np.argwhere(v[:, 0] == i_row), axis=0)
                # Increase row value of all positions above (index < i_row) by 1
                v[np.argwhere(v[:, 0] < i_row), 0] += 1
                # Update positions
                self.positions[k] = v
            filled_per_row = np.sum(self.get_board() > 0, axis=1)
            cleared += 1
        # Determine score depending on level and number of lines cleared
        level = self.lines // 10
        scores = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}
        self.score += scores[cleared] * (level + 1)
        self.lines += cleared
        # Reward is equal to cleared rows if cleared > 0
        if cleared > 0:
            reward = cleared 
        # Else: reward is equal to density of filled cells
        else:
            reward = self.density()
        # Generate new tetrimino and print environment if gui is enabled
        self.new_tetrimino()
        self.show()
        # Return observation, reward, done (=False), and all invalid actions
        return self.obs(), reward, False, self.invalid()