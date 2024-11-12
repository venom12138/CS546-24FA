'''
dict: key:{'id': {'answer': str,
                    'num_attempts': int,
                    'guesses': [word1, word2, word3, word4, word5, word6], 
                    'responses': [res1, res2, res3, res4, res5, res6],
                    'figures': [id_0, id_1, id_2, id_3, id_4, id_5, id_6]}}
'''
from solvers.wordle_solver import WORDS, solver
from envs.wordle import WordleEnv
import numpy as np
import os

