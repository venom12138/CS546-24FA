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

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARKGRAY = (40, 40, 40)


@dataclass
class Base:
    cell_size: int = 20
    cell_width: int = 12
    cell_height: int = 12
    window_width = cell_size * cell_width
    window_height = cell_size * cell_height

    @staticmethod
    def node_add(node_a: Tuple[int, int], node_b: Tuple[int, int]):
        result: Tuple[int, int] = tuple(map(add, node_a, node_b))
        return result

    @staticmethod
    def node_sub(node_a: Tuple[int, int], node_b: Tuple[int, int]):
        result: Tuple[int, int] = tuple(map(sub, node_a, node_b))
        return result

    @staticmethod
    def mean(l):
        return round(sum(l) / len(l), 4)

class Apple(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.location = None

    def refresh(self, snake):
        """
        Generate a new apple
        """
        available_positions = set(product(range(self.cell_width - 1), range(self.cell_height - 1))) - set(snake.body)

        # If there's no available node for new apple, it reaches the perfect solution. Don't draw the apple then.
        location = random.sample(available_positions, 1)[0] if available_positions else (-1, -1)

        self.location = location


class Snake(Base):
    def __init__(self, initial_length: int = 3, body: list = None, **kwargs):
        """
        :param initial_length: The initial length of the snake
        :param body: Optional. Specifying an initial snake body
        """
        super().__init__(**kwargs)
        self.initial_length = initial_length
        self.score = 0
        self.is_dead = False
        self.eaten = False

        # last_direction is only used for human player, giving it a default direction when game starts
        self.last_direction = (-1, 0)

        if body:
            self.body = body
        else:
            if not 0 < initial_length < self.cell_width:
                raise ValueError(f"Initial_length should fall in (0, {self.cell_width})")

            start_x = self.cell_width // 2
            start_y = self.cell_height // 2

            start_body_x = [start_x] * initial_length
            start_body_y = range(start_y, start_y - initial_length, -1)

            self.body = list(zip(start_body_x, start_body_y))

    def get_head(self):
        return self.body[-1]

    def dead_checking(self, head, check=False):
        """
        Check if the snake is dead
        :param check: if check is True, only return the checking result without updating snake.is_dead
        :return: Boolean
        """
        x, y = head
        if not 0 <= x < self.cell_width or not 0 <= y < self.cell_height or head in self.body[1:]:
            if not check:
                self.is_dead = True
            return True
        return False

    def cut_tail(self):
        self.body.pop(0)

    def move(self, new_head: tuple, apple: Apple):
        """
        Given the location of apple, decide if the apple is eaten (same location as the snake's head)
        :param new_head: (new_head_x, new_head_y)
        :param apple: Apple instance
        :return: Boolean. Whether the apple is eaten.
        """
        if new_head is None:
            self.is_dead = True
            return

        if self.dead_checking(head=new_head):
            return

        self.last_direction = self.node_sub(new_head, self.get_head())

        # make the move
        self.body.append(new_head)

        # if the snake eats the apple, score adds 1
        if self.get_head() == apple.location:
            self.eaten = True
            self.score += 1
        # Otherwise, cut the tail so that snake moves forward without growing
        else:
            self.eaten = False
            self.cut_tail()
