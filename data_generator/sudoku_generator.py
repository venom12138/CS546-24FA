import sys
sys.path.append('./')
from solvers.sudoku_solver import sudoku_solver_with_meaningful_saving
from envs.sudoku import SudokuEnv

env = SudokuEnv()
env.reset()
if sudoku_solver_with_meaningful_saving(env):
    print("Puzzle solved successfully!")
else:
    print("No solution exists.")