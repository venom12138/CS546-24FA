'''
dict: key:{'id': {'num_attempts': int,
                'actions': [a_0, a_1, a_2, ..., a_{N-1}], 
                'states': [s_0, s_1, s_2, ..., s_N],
                'figures': [id_0, id_1, id_2, ..., id_N]}}
'''
import gym
import gym_sokoban
import time
from PIL import Image
import numpy as np
import os
from multiprocessing import Process, Pool
import sys
sys.path.append('./')
from solvers.sokoban_solver.sokoban import solve_sokoban

# Before you can make a Sokoban Environment you need to call:
# cd gym_sokoban, pip install -e .
# import gym_sokoban
# This import statement registers all Sokoban environments
# provided by this package

NUM_TRAJS = 500
SAVE_PATH = 'trajectories/sokoban_data'
os.makedirs(SAVE_PATH, exist_ok=True)
env_name = 'Sokoban-v0'
env = gym.make(env_name)

ACTION_LOOKUP = env.unwrapped.get_action_lookup()

def gen_trajs(start_idx, end_idx):
    env = gym.make(env_name)
    all_trajs = {}
    i = start_idx
    while i < end_idx:
        env.reset()
        solver_action = solve_sokoban(env)
        os.makedirs(f"{SAVE_PATH}/traj_{i}", exist_ok=True)
        if solver_action is None:
            print('this sokoban is unsolvable')
            env.close()
            env = gym.make(env_name)
        else:
            '''
            symbol_map = {
                0: '#',  # Wall
                1: ' ',  # Movable space
                2: '.',  # Destination
                3: 'X',  # box on target
                4: 'B',  # box not on target
                5: '&',  # player
            }
            '''
            state = env.room_state
            img = env.render(mode='rgb_array')
            image = Image.fromarray(img)
            image.save(f"{SAVE_PATH}/traj_{i}/0.png")
            
            traj_data = {'figures': [f"{SAVE_PATH}/traj_{i}/0.png"],
                        'states': [state],
                        'actions': []}
            
            for t in range(len(solver_action)):
                # solver approach
                action = solver_action[t]
                observation, reward, done, info = env.step(action)
                
                state = env.room_state
                img = env.render(mode='rgb_array')
                # save this image as a file
                image = Image.fromarray(img)
                image.save(f"{SAVE_PATH}/traj_{i}/{t+1}.png")
                
                traj_data['states'].append(state)
                traj_data['figures'].append(f"{SAVE_PATH}/traj_{i}/{t+1}.png")
                traj_data['actions'].append(ACTION_LOOKUP[action])
                
                # print(ACTION_LOOKUP[action], reward, done, info)
                if done:
                    print("Episode {} finished after {} timesteps".format(i, t+1))
                    traj_data['num_attempts'] = t + 1
                    break
            all_trajs[f"id_{i}"] = traj_data
            i += 1
    env.close()
    return all_trajs


# multiprocessing generate data
NUM_PROCS = 10
args_list = [(i * NUM_TRAJS // NUM_PROCS, (i + 1) * NUM_TRAJS // NUM_PROCS) for i in range(NUM_PROCS)]

with Pool(NUM_PROCS) as pool:
    results = [pool.apply_async(gen_trajs, args) for args in args_list]
    output = [r.get() for r in results]

all_trajs = {}
for out in output:
    all_trajs.update(out)

np.savez(f"{SAVE_PATH}/data.npz", **all_trajs)
