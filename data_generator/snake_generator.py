from PIL import Image, ImageDraw
import numpy as np
import os
import sys
sys.path.append('./')
import random
import time
from operator import add, sub
from dataclasses import dataclass
from itertools import product
from typing import Tuple
from multiprocessing import Pool, cpu_count

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARKGRAY = (40, 40, 40)

# Assuming that solvers.snake is your module with the necessary classes
from solvers.snake_solver import Base, Apple, Snake, Mixed

NUM_TRAJS = 100
SAVE_PATH = 'trajectories/snake_data'
os.makedirs(SAVE_PATH, exist_ok=True)

@dataclass
class SnakeGame(Base):

    def __init__(self, trial_num, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.step_data = []  # List to store step data
        self.trial_num = trial_num
        self.trial_path = os.path.join(SAVE_PATH, f'trial_{self.trial_num}')
        os.makedirs(self.trial_path, exist_ok=True)

    def launch(self):
        self.game()

    def game(self):
        snake = Snake(**self.kwargs)
        apple = Apple(**self.kwargs)
        apple.refresh(snake=snake)

        step_time = []
        step_count = 0  # To name the saved images
        actions = []  # List to store actions

        for _ in range(100):
            start_time = time.time()

            # Use the AI player (you can switch to other strategies if needed)
            new_head = Mixed(snake=snake, apple=apple, **self.kwargs).run_mixed()

            end_time = time.time()
            move_time = end_time - start_time
            step_time.append(move_time)

            # Determine the movement direction
            movement = self.get_movement_direction(snake.get_head(), new_head)
            actions.append(movement)

            snake.move(new_head=new_head, apple=apple)

            if snake.is_dead:
                print(f"Trial {self.trial_num}: Snake is dead.")
                break
            elif snake.eaten:
                apple.refresh(snake=snake)
            
            if apple.location[0]<0 and apple.location[1] < 0:
                break

            if snake.score + snake.initial_length >= self.cell_width * self.cell_height - 1:
                print(f"Trial {self.trial_num}: Snake has filled the board!")
                break

            # Generate image and save
            image = self.render_image(snake.body, apple.location)
            image_path = os.path.join(self.trial_path, f'step_{step_count}.png')
            image.save(image_path)

            # Get text description and save
            text_description = self.get_text_description(snake.body, apple.location)
            text_path = os.path.join(self.trial_path, f'step_{step_count}.txt')
            with open(text_path, 'w') as f:
                f.write(text_description)

            step_count += 1

        # Save all actions to a text file
        actions_path = os.path.join(self.trial_path, 'actions.txt')
        with open(actions_path, 'w') as f:
            f.write('\n'.join(actions))

        print(f"Trial {self.trial_num}: Score: {snake.score}")
        print(f"Trial {self.trial_num}: Mean step time: {self.mean(step_time)}")

    @staticmethod
    def get_movement_direction(old_head, new_head):
        dx = new_head[0] - old_head[0]
        dy = new_head[1] - old_head[1]
        if dx == 1 and dy == 0:
            return 'RIGHT'
        elif dx == -1 and dy == 0:
            return 'LEFT'
        elif dx == 0 and dy == 1:
            return 'DOWN'
        elif dx == 0 and dy == -1:
            return 'UP'
        else:
            return 'UNKNOWN'

    def render_image(self, snake_body, apple_location):
        # Create an image of the board
        img_width = self.cell_width * self.cell_size
        img_height = self.cell_height * self.cell_size
        image = Image.new('RGB', (img_width, img_height), 'black')
        draw = ImageDraw.Draw(image)

        # Draw grid
        for x in range(0, img_width, self.cell_size):
            draw.line((x, 0, x, img_height), fill=DARKGRAY)
        for y in range(0, img_height, self.cell_size):
            draw.line((0, y, img_width, y), fill=DARKGRAY)

        # Draw snake body
        for x, y in snake_body[:-1]:
            rect = [
                x * self.cell_size,
                y * self.cell_size,
                (x + 1) * self.cell_size - 1,
                (y + 1) * self.cell_size - 1
            ]
            draw.rectangle(rect, fill=WHITE)

        # Draw snake head in green
        x, y = snake_body[-1]
        rect = [
            x * self.cell_size,
            y * self.cell_size,
            (x + 1) * self.cell_size - 1,
            (y + 1) * self.cell_size - 1
        ]
        draw.rectangle(rect, fill=GREEN)

        # Draw apple
        x, y = apple_location
        rect = [
            x * self.cell_size,
            y * self.cell_size,
            (x + 1) * self.cell_size - 1,
            (y + 1) * self.cell_size - 1
        ]
        draw.rectangle(rect, fill=RED)

        return image

    def get_text_description(self, snake_body, apple_location):
        grid = [['x' for _ in range(self.cell_width)] for _ in range(self.cell_height)]
        # Place apple
        ax, ay = apple_location
        grid[ay][ax] = 'o'
        # Place snake body with numbers, 0 as head
        # Reverse the snake_body to assign numbers from 0 (head) to N-1 (tail)
        for idx in range(len(snake_body)-1):
            (x1, y1) = snake_body[idx]
            (x0, y0) = snake_body[idx+1]
            if x1 == x0 and y1 < y0:
                grid[y1][x1] = "\\"
            if x1 == x0 and y1 > y0:
                grid[y1][x1] = '/'
            if x1 < x0 and y1 == y0:
                grid[y1][x1] = '>'
            if x1 > x0 and y1 == y0:
                grid[y1][x1] = '<'
        x, y = snake_body[-1]
        grid[y][x] = 'H'
        # Convert grid to text description
        text_description = '\n'.join(''.join(row) for row in grid)
        return text_description

    @staticmethod
    def mean(l):
        return round(sum(l) / len(l), 4)

# Function to run a single trial
def run_trial(trial_num):
    game = SnakeGame(trial_num=trial_num)
    game.launch()

# Main execution
if __name__ == "__main__":
    # Determine the number of processes to use (2x number of CPU cores or NUM_TRAJS, whichever is smaller)
    num_processes = min(NUM_TRAJS, cpu_count() * 2)
    np.random.seed(34332940)
    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Map the run_trial function to the number of trials
        pool.map(run_trial, range(NUM_TRAJS))
