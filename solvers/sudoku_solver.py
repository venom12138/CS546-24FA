import numpy as np
import gym
from gym import spaces
from PIL import Image, ImageDraw, ImageFont
import random

import os
import numpy as np

def sudoku_solver_with_meaningful_saving(env, output_folder="sudoku_steps"):
    """
    Sudoku solver that saves only meaningful steps as images.

    Args:
        env (SudokuEnv): An instance of the SudokuEnv class.
        output_folder (str): Directory to save unique intermediate images.

    Returns:
        bool: True if the puzzle is solved, False otherwise.
    """
    arr = env.current_puzzle
    pos = {}
    rem = {}
    graph = {}

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    step_count = 0
    last_saved_state = None

    def save_step():
        """Save the current puzzle state as an image if it's meaningful."""
        nonlocal step_count, last_saved_state
        current_state = env.current_puzzle.copy()

        # Save only if there is progress (more filled cells than the last state)
        if (
            last_saved_state is None
            or np.sum(current_state == 0) < np.sum(last_saved_state == 0)
        ):
            import pdb; pdb.set_trace()
            last_saved_state = current_state
            img = env.render()
            np.savetxt(os.path.join(output_folder, f"step_{step_count:03d}.txt"), current_state, fmt="%d")
            img.save(os.path.join(output_folder, f"step_{step_count:03d}.png"))
            step_count += 1

    def is_safe(x, y):
        """Check if placing a number at position (x, y) is safe."""
        key = arr[x][y]
        for i in range(0, 9):
            if i != y and arr[x][i] == key:
                return False
            if i != x and arr[i][y] == key:
                return False

        r_start = int(x / 3) * 3
        r_end = r_start + 3

        c_start = int(y / 3) * 3
        c_end = c_start + 3

        for i in range(r_start, r_end):
            for j in range(c_start, c_end):
                if i != x and j != y and arr[i][j] == key:
                    return False
        return True

    def fill_matrix(k, keys, r, rows):
        """Fill the matrix using the graph-based approach with updates at each step."""
        for c in graph[keys[k]][rows[r]]:
            if arr[rows[r]][c] > 0:
                continue
            arr[rows[r]][c] = keys[k]
            env.current_puzzle = arr.copy()
            save_step()  # Save only if meaningful progress is made
            if is_safe(rows[r], c):
                if r < len(rows) - 1:
                    if fill_matrix(k, keys, r + 1, rows):
                        return True
                    else:
                        arr[rows[r]][c] = 0  # Backtrack
                        env.current_puzzle = arr.copy()
                        continue
                else:
                    if k < len(keys) - 1:
                        if fill_matrix(k + 1, keys, 0, list(graph[keys[k + 1]].keys())):
                            return True
                        else:
                            arr[rows[r]][c] = 0  # Backtrack
                            env.current_puzzle = arr.copy()
                            continue
                    return True
            arr[rows[r]][c] = 0  # Backtrack
            env.current_puzzle = arr.copy()
        return False

    def build_pos_and_rem():
        """Build the pos and rem dictionaries to track remaining numbers and positions."""
        for i in range(0, 9):
            for j in range(0, 9):
                if arr[i][j] > 0:
                    if arr[i][j] not in pos:
                        pos[arr[i][j]] = []
                    pos[arr[i][j]].append([i, j])
                    if arr[i][j] not in rem:
                        rem[arr[i][j]] = 9
                    rem[arr[i][j]] -= 1

        for i in range(1, 10):
            if i not in pos:
                pos[i] = []
            if i not in rem:
                rem[i] = 9

    def build_graph():
        """Build the graph to track possible positions for each number."""
        for k, v in pos.items():
            if k not in graph:
                graph[k] = {}

            row = list(range(0, 9))
            col = list(range(0, 9))

            for cord in v:
                row.remove(cord[0])
                col.remove(cord[1])

            if len(row) == 0 or len(col) == 0:
                continue

            for r in row:
                for c in col:
                    if arr[r][c] == 0:
                        if r not in graph[k]:
                            graph[k][r] = []
                        graph[k][r].append(c)

    # Build dictionaries and graph
    build_pos_and_rem()
    rem = {k: v for k, v in sorted(rem.items(), key=lambda item: item[1])}
    build_graph()

    # Solve the Sudoku puzzle
    key_s = list(rem.keys())
    return fill_matrix(0, key_s, 0, list(graph[key_s[0]].keys()))

# Initialize and solve the Sudoku puzzle
# env = SudokuEnv()
# env.reset()
# if sudoku_solver_with_meaningful_saving(env):
#     print("Puzzle solved successfully!")
# else:
#     print("No solution exists.")
