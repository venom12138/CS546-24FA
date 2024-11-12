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

CANDIDATE_WORDS = list(WORDS)
NUM_TRAJS = 1000
SAVE_PATH = 'trajectories/wordle_data'
os.makedirs(SAVE_PATH, exist_ok=True)
env = WordleEnv(word_list=CANDIDATE_WORDS, max_attempts=6)
all_trajs = {}
for i in range(NUM_TRAJS):
    observation = env.reset()
    img = env.render(mode="RGB")
    os.makedirs(f"{SAVE_PATH}/traj_{i}", exist_ok=True)
    img.save(f"{SAVE_PATH}/traj_{i}/0.png")
    answer = env.target_word
    guesses, responses = solver(answer)
    traj_data = {'answer': answer,
                 'figures': [f"{SAVE_PATH}/traj_{i}/0.png"],}
    for j in range(1, 1+env.max_attempts):
        action = guesses.pop(0)
        observation, reward, done, info = env.step(action)
        img = env.render(mode="RGB")
        img.save(f"{SAVE_PATH}/traj_{i}/{j}.png")
        traj_data['figures'].append(f"{SAVE_PATH}/traj_{i}/{j}.png")
        if done:
            traj_data['guesses'] = observation['guesses']
            traj_data['responses'] = observation['feedbacks']
            traj_data['num_attemps'] = j
            break
    all_trajs[f"id_{i}"] = traj_data

# save as json file
import json
with open(f"{SAVE_PATH}/data.json", 'w') as f:
    json.dump(all_trajs, f)