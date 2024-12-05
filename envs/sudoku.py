import numpy as np
import gym
from gym import spaces
from PIL import Image, ImageDraw, ImageFont
import random

class SudokuEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, puzzle=None, cell_size=60, line_width=2, subgrid_line_width=4, font_size=48, num_clues=30):
        """
        Initialize the Sudoku environment.

        Args:
            puzzle (np.ndarray or None): A 9x9 numpy array representing the Sudoku puzzle. 
                                         0 denotes empty cells. If None, a random puzzle is generated.
            cell_size (int): Size of each cell in rendered image.
            line_width (int): Width of normal grid lines.
            subgrid_line_width (int): Width of thicker lines separating 3x3 subgrids.
            font_size (int): Font size for numbers in rendering.
            num_clues (int): Approximate number of clues to remain in the randomly generated puzzle.
                             The final number of clues may vary slightly.
        """
        super(SudokuEnv, self).__init__()
        
        self.action_space = spaces.MultiDiscrete([9, 9, 9])
        self.observation_space = spaces.Box(low=0, high=9, shape=(9,9), dtype=np.int32)
        
        self.cell_size = cell_size
        self.line_width = line_width
        self.subgrid_line_width = subgrid_line_width
        self.font_size = font_size
        self.num_clues = num_clues

        self.board_size = cell_size * 9
        
        # If puzzle is None, generate a random puzzle
        if puzzle is None:
            self.initial_puzzle = self._generate_puzzle(num_clues=self.num_clues)
        else:
            assert puzzle.shape == (9,9), "Puzzle must be a 9x9 grid"
            self.initial_puzzle = puzzle.copy()

        self.current_puzzle = None
        self.done = False
        
        try:
            self.font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # Fallback if arial.ttf is not available
            self.font = ImageFont.load_default(size=font_size)

    def reset(self):
        self.current_puzzle = self.initial_puzzle.copy()
        self.done = False
        return self._get_observation()

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, {}
        
        row, col, val = action
        reward = 0

        if self.current_puzzle[row, col] != 0:
            # Cell already filled
            reward = -1
        else:
            # Check legality
            if self._is_legal(row, col, val):
                self.current_puzzle[row, col] = val
                reward = 1
            else:
                reward = -1

        if self._is_solved():
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def render(self, mode='human'):
        img = Image.new('RGB', (self.board_size, self.board_size), 'white')
        draw = ImageDraw.Draw(img)

        # Draw the grid lines
        for i in range(10):
            line_thickness = self.subgrid_line_width if i % 3 == 0 else self.line_width
            x = i * self.cell_size
            draw.line([(x, 0), (x, self.board_size)], fill='black', width=line_thickness)
            y = i * self.cell_size
            draw.line([(0, y), (self.board_size, y)], fill='black', width=line_thickness)

        # Draw the numbers
        for i in range(9):
            for j in range(9):
                val = self.current_puzzle[i, j]
                if val != 0:
                    text = str(val)
                    left, top, right, bottom = self.font.getbbox(text)
                    w = right - left
                    h = bottom - top
                    cell_x = j * self.cell_size
                    cell_y = i * self.cell_size
                    x_pos = cell_x + (self.cell_size - w) / 2
                    # Adjust y_pos so the number is centered vertically, or at least nicely placed
                    # Currently just placing at top; let's center in cell
                    y_pos = cell_y
                    draw.text((x_pos, y_pos), text, fill='black', font=self.font)

        return img

    def _get_observation(self):
        return self.current_puzzle.copy()

    def _is_legal(self, row, col, val):
        # Check row
        if val in self.current_puzzle[row, :]:
            return False
        # Check column
        if val in self.current_puzzle[:, col]:
            return False
        # Check 3x3 sub-grid
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        if val in self.current_puzzle[box_row:box_row+3, box_col:box_col+3]:
            return False
        return True

    def _is_solved(self):
        if np.any(self.current_puzzle == 0):
            return False
        for i in range(9):
            row_vals = self.current_puzzle[i, :]
            col_vals = self.current_puzzle[:, i]
            if (len(set(row_vals)) != 9) or (len(set(col_vals)) != 9):
                return False
        for box_row in range(0,9,3):
            for box_col in range(0,9,3):
                box = self.current_puzzle[box_row:box_row+3, box_col:box_col+3].flatten()
                if len(set(box)) != 9:
                    return False
        return True

    def _check_safe(self, grid, row, col, val):
        # Check row
        if val in grid[row, :]:
            return False
        # Check column
        if val in grid[:, col]:
            return False
        # Check 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        if val in grid[box_row:box_row+3, box_col:box_col+3]:
            return False
        return True

    def _solve_randomly(self, grid):
        # Find the next empty cell (marked by 0)
        indices = np.where(grid == 0)
        if len(indices[0]) == 0:
            # No empty cells, solution found
            return True
        
        row = indices[0][0]
        col = indices[1][0]

        # Try digits 1 to 9 in random order
        digits = list(range(1, 10))
        random.shuffle(digits)
        for val in digits:
            if self._check_safe(grid, row, col, val):
                grid[row, col] = val
                if self._solve_randomly(grid):
                    return True
                # Backtrack
                grid[row, col] = 0
        return False

    def generate_random_solution(self,):
        """
        Generate a completely random valid Sudoku solution using backtracking.
        
        Returns:
            solution (np.ndarray): A 9x9 numpy array representing a valid Sudoku solution.
        """
        solved = False
        while not solved:
            grid = np.zeros((9, 9), dtype=np.int32)
            solved = self._solve_randomly(grid)
        return grid
    
    def _generate_puzzle(self, num_clues=30):
        """
        Generate a random Sudoku puzzle by:
        1. Starting from a known valid Sudoku solution.
        2. Applying random transformations that preserve validity.
        3. Removing digits to achieve the desired number of clues.

        This approach yields more diverse puzzles than the fixed solution approach.

        Args:
            num_clues (int): The desired number of clues to keep. The final puzzle may not 
                            have exactly this number, but it will be close.

        Returns:
            puzzle (np.ndarray): A 9x9 Sudoku puzzle with given number of clues.
        """
        # A known valid Sudoku solution
        solution = self.generate_random_solution()
        # np.array([
        #     [5,3,4,6,7,8,9,1,2],
        #     [6,7,2,1,9,5,3,4,8],
        #     [1,9,8,3,4,2,5,6,7],
        #     [8,5,9,7,6,1,4,2,3],
        #     [4,2,6,8,5,3,7,9,1],
        #     [7,1,3,9,2,4,8,5,6],
        #     [9,6,1,5,3,7,2,8,4],
        #     [2,8,7,4,1,9,6,3,5],
        #     [3,4,5,2,8,6,1,7,9]
        # ], dtype=np.int32)

        puzzle = solution.copy()

        # Apply random transformations

        # 1. Randomly swap entire row bands
        # A row band is a group of three rows: [0-2], [3-5], [6-8]
        row_bands = [0,1,2]  # band indices
        np.random.shuffle(row_bands)
        new_order = []
        for band in row_bands:
            new_order.extend([band*3, band*3+1, band*3+2])
        puzzle = puzzle[new_order, :]

        # 2. Randomly swap entire column bands
        col_bands = [0,1,2]
        np.random.shuffle(col_bands)
        new_order = []
        for band in col_bands:
            new_order.extend([band*3, band*3+1, band*3+2])
        puzzle = puzzle[:, new_order]

        # 3. Randomly swap rows within each band
        for band_start in [0,3,6]:
            rows = [band_start, band_start+1, band_start+2]
            np.random.shuffle(rows)
            puzzle[band_start:band_start+3, :] = puzzle[rows, :]

        # 4. Randomly swap columns within each band
        for band_start in [0,3,6]:
            cols = [band_start, band_start+1, band_start+2]
            np.random.shuffle(cols)
            puzzle[:, band_start:band_start+3] = puzzle[:, cols]

        # 5. Randomly permute the digits 1-9
        # Create a mapping of digits: 
        # for example {1:9, 2:3, ...} to shuffle digits
        digits = [1,2,3,4,5,6,7,8,9]
        np.random.shuffle(digits)
        mapping = {original: new for original, new in zip(range(1,10), digits)}
        # Apply the mapping
        for r in range(9):
            for c in range(9):
                val = puzzle[r,c]
                if val != 0:
                    puzzle[r,c] = mapping[val]

        # Now remove random cells to achieve approximately num_clues clues
        # Generate a list of all cell positions
        cells = [(r,c) for r in range(9) for c in range(9)]
        random.shuffle(cells)

        cells_to_remove = 81 - num_clues
        for i in range(cells_to_remove):
            r, c = cells[i]
            puzzle[r, c] = 0

        return puzzle

if __name__ == "__main__":
    env = SudokuEnv()
    env.reset()
    img = env.render()
    img.show()

    img.save('./sudoku.png')  # Displays the board if you're running this locally
