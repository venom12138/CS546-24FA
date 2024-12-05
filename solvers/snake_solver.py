#!/usr/bin/python
# -*-coding: utf-8 -*-

import contextlib
import random
import sys
import time
from operator import add, sub
from dataclasses import dataclass
from itertools import product
from typing import Tuple

with contextlib.redirect_stdout(None):
    import pygame
    from pygame.locals import *
from heapq import *

from envs.snake import Base, Apple, Snake

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARKGRAY = (40, 40, 40)

def heuristic(start, goal):
    return (start[0] - goal[0])**2 + (start[1] - goal[1])**2


class Player(Base):
    def __init__(self, snake: Snake, apple: Apple, **kwargs):
        """
        :param snake: Snake instance
        :param apple: Apple instance
        """
        super().__init__(**kwargs)
        self.snake = snake
        self.apple = apple

    def _get_neighbors(self, node):
        """
        fetch and yield the four neighbours of a node
        :param node: (node_x, node_y)
        """
        for diff in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            yield self.node_add(node, diff)

    @staticmethod
    def is_node_in_queue(node: tuple, queue: iter):
        """
        Check if element is in a nested list
        """
        return any(node in sublist for sublist in queue)

    def is_invalid_move(self, node: tuple, snake: Snake):
        """
        Similar to dead_checking, this method checks if a given node is a valid move
        :return: Boolean
        """
        x, y = node
        if not 0 <= x < self.cell_width or not 0 <= y < self.cell_height or node in snake.body:
            return True
        return False


class BFS(Player):
    def __init__(self, snake: Snake, apple: Apple, **kwargs):
        """
        :param snake: Snake instance
        :param apple: Apple instance
        """
        super().__init__(snake=snake, apple=apple, **kwargs)

    def run_bfs(self):
        """
        Run BFS searching and return the full path of best way to apple from BFS searching
        """
        queue = [[self.snake.get_head()]]

        while queue:
            path = queue[0]
            future_head = path[-1]

            # If snake eats the apple, return the next move after snake's head
            if future_head == self.apple.location:
                return path

            for next_node in self._get_neighbors(future_head):
                if (
                    self.is_invalid_move(node=next_node, snake=self.snake)
                    or self.is_node_in_queue(node=next_node, queue=queue)
                ):
                    continue
                new_path = list(path)
                new_path.append(next_node)
                queue.append(new_path)

            queue.pop(0)

    def next_node(self):
        """
        Run the BFS searching and return the next move in this path
        """
        path = self.run_bfs()
        return path[1]


class LongestPath(BFS):
    """
    Given shortest path, change it to the longest path
    """

    def __init__(self, snake: Snake, apple: Apple, **kwargs):
        """
        :param snake: Snake instance
        :param apple: Apple instance
        """
        super().__init__(snake=snake, apple=apple, **kwargs)
        self.kwargs = kwargs

    def run_longest(self):
        """
        For every move, check if it could be replace with three equivalent moves.
        For example, for snake moving one step left, check if moving up, left, and down is valid. If yes, replace the
        move with equivalent longer move. Start this over until no move can be replaced.
        """
        path = self.run_bfs()

        # print(f'longest path initial result: {path}')

        if path is None:
            # print(f"Has no Longest path")
            return

        i = 0
        while True:
            try:
                direction = self.node_sub(path[i], path[i + 1])
            except IndexError:
                break

            # Build a dummy snake with body and longest path for checking if node replacement is valid
            snake_path = Snake(body=self.snake.body + path[1:], **self.kwargs)

            # up -> left, up, right
            # down -> right, down, left
            # left -> up, left, down
            # right -> down, right, up
            for neibhour in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if direction == neibhour:
                    x, y = neibhour
                    diff = (y, x) if x != 0 else (-y, x)

                    extra_node_1 = self.node_add(path[i], diff)
                    extra_node_2 = self.node_add(path[i + 1], diff)

                    if snake_path.dead_checking(head=extra_node_1) or snake_path.dead_checking(head=extra_node_2):
                        i += 1
                    else:
                        # Add replacement nodes
                        path[i + 1:i + 1] = [extra_node_1, extra_node_2]
                    break

        # Exclude the first node, which is same to snake's head
        return path[1:]


class Fowardcheck(Player):
    def __init__(self, snake: Snake, apple: Apple, **kwargs):
        """
        :param snake: Snake instance
        :param apple: Apple instance
        """
        super().__init__(snake=snake, apple=apple, **kwargs)
        self.kwargs = kwargs

    def run_forwardcheck(self):
        bfs = BFS(snake=self.snake, apple=self.apple, **self.kwargs)

        path = bfs.run_bfs()

        print("trying BFS")

        if path is None:
            snake_tail = Apple()
            snake_tail.location = self.snake.body[0]
            snake = Snake(body=self.snake.body[1:])
            longest_path = LongestPath(snake=snake, apple=snake_tail, **self.kwargs).run_longest()
            next_node = longest_path[0]
            # print("BFS not reachable, trying head to tail")
            # print(next_node)
            return next_node

        length = len(self.snake.body)
        virtual_snake_body = (self.snake.body + path[1:])[-length:]
        virtual_snake_tail = Apple()
        virtual_snake_tail.location = (self.snake.body + path[1:])[-length - 1]
        virtual_snake = Snake(body=virtual_snake_body)
        virtual_snake_longest = LongestPath(snake=virtual_snake, apple=virtual_snake_tail, **self.kwargs)
        virtual_snake_longest_path = virtual_snake_longest.run_longest()
        if virtual_snake_longest_path is None:
            snake_tail = Apple()
            snake_tail.location = self.snake.body[0]
            snake = Snake(body=self.snake.body[1:])
            longest_path = LongestPath(snake=snake, apple=snake_tail, **self.kwargs).run_longest()
            next_node = longest_path[0]
            # print("virtual snake not reachable, trying head to tail")
            # print(next_node)
            return next_node
        else:
            # print("BFS accepted")
            return path[1]


class Mixed(Player):
    def __init__(self, snake: Snake, apple: Apple, **kwargs):
        """
        :param snake: Snake instance
        :param apple: Apple instance
        """
        super().__init__(snake=snake, apple=apple, **kwargs)
        self.kwargs = kwargs

    def escape(self):
        head = self.snake.get_head()
        largest_neibhour_apple_distance = 0
        newhead = None
        for diff in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            neibhour = self.node_add(head, diff)

            if self.snake.dead_checking(head=neibhour, check=True):
                continue

            neibhour_apple_distance = (
                abs(neibhour[0] - self.apple.location[0]) + abs(neibhour[1] - self.apple.location[1])
            )
            # Find the neibhour which has greatest Manhattan distance to apple and has path to tail
            if largest_neibhour_apple_distance < neibhour_apple_distance:
                snake_tail = Apple()
                snake_tail.location = self.snake.body[1]
                # Create a virtual snake with a neibhour as head, to see if it has a way to its tail,
                # thus remove two nodes from body: one for moving one step forward, one for avoiding dead checking
                snake = Snake(body=self.snake.body[2:] + [neibhour])
                bfs = BFS(snake=snake, apple=snake_tail, **self.kwargs)
                path = bfs.run_bfs()
                if path is None:
                    continue
                largest_neibhour_apple_distance = neibhour_apple_distance
                newhead = neibhour
        return newhead

    def run_mixed(self):
        """
        Mixed strategy
        """
        bfs = BFS(snake=self.snake, apple=self.apple, **self.kwargs)

        path = bfs.run_bfs()

        # If the snake does not have the path to apple, try to follow its tail to escape
        if path is None:
            return self.escape()

        # Send a virtual snake to see when it reaches the apple, does it still have a path to its own tail, to keep it
        # alive
        length = len(self.snake.body)
        virtual_snake_body = (self.snake.body + path[1:])[-length:]
        virtual_snake_tail = Apple()
        virtual_snake_tail.location = (self.snake.body + path[1:])[-length - 1]
        virtual_snake = Snake(body=virtual_snake_body)
        virtual_snake_longest = BFS(snake=virtual_snake, apple=virtual_snake_tail, **self.kwargs)
        virtual_snake_longest_path = virtual_snake_longest.run_bfs()
        if virtual_snake_longest_path is None:
            return self.escape()
        else:
            return path[1]


class Astar(Player):
    def __init__(self, snake: Snake, apple: Apple, **kwargs):
        """
        :param snake: Snake instance
        :param apple: Apple instance
        """
        super().__init__(snake=snake, apple=apple, **kwargs)
        self.kwargs = kwargs

    def run_astar(self):
        came_from = {}
        close_list = set()
        goal = self.apple.location
        start = self.snake.get_head()
        dummy_snake = Snake(body=self.snake.body)
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)]
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        open_list = [(fscore[start], start)]
        print(start, goal, open_list)
        while open_list:
            current = min(open_list, key=lambda x: x[0])[1]
            open_list.pop(0)
            print(current)
            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                    print(data)
                return data[-1]

            close_list.add(current)

            for neighbor in neighbors:
                neighbor_node = self.node_add(current, neighbor)

                if dummy_snake.dead_checking(head=neighbor_node) or neighbor_node in close_list:
                    continue
                if sum(map(abs, self.node_sub(current, neighbor_node))) == 2:
                    diff = self.node_sub(current, neighbor_node)
                    if dummy_snake.dead_checking(head=self.node_add(neighbor_node, (0, diff[1]))
                                                 ) or self.node_add(neighbor_node, (0, diff[1])) in close_list:
                        continue
                    elif dummy_snake.dead_checking(head=self.node_add(neighbor_node, (diff[0], 0))
                                                   ) or self.node_add(neighbor_node, (diff[0], 0)) in close_list:
                        continue
                tentative_gscore = gscore[current] + heuristic(current, neighbor_node)
                if tentative_gscore < gscore.get(neighbor_node, 0) or neighbor_node not in [i[1] for i in open_list]:
                    gscore[neighbor_node] = tentative_gscore
                    fscore[neighbor_node] = tentative_gscore + heuristic(neighbor_node, goal)
                    open_list.append((fscore[neighbor_node], neighbor_node))
                    came_from[neighbor_node] = current


class Human(Player):
    def __init__(self, snake: Snake, apple: Apple, **kwargs):
        """
        :param snake: Snake instance
        :param apple: Apple instance
        """
        super().__init__(snake=snake, apple=apple, **kwargs)

    def run(self):
        for event in pygame.event.get():  # event handling loop
            if event.type == KEYDOWN:
                if (event.key == K_LEFT or event.key == K_a) and self.snake.last_direction != (1, 0):
                    diff = (-1, 0)  # left
                elif (event.key == K_RIGHT or event.key == K_d) and self.snake.last_direction != (-1, 0):
                    diff = (1, 0)  # right
                elif (event.key == K_UP or event.key == K_w) and self.snake.last_direction != (0, 1):
                    diff = (0, -1)  # up
                elif (event.key == K_DOWN or event.key == K_s) and self.snake.last_direction != (0, -1):
                    diff = (0, 1)  # down
                else:
                    break
                return self.node_add(self.snake.get_head(), diff)
        # If no button is pressed down, follow previou direction
        return self.node_add(self.snake.get_head(), self.snake.last_direction)