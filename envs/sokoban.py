import gym
import gym_sokoban
import time
from PIL import Image
import numpy as np

# Before you can make a Sokoban Environment you need to call:
# cd gym_sokoban, pip install -e .
# import gym_sokoban
# This import statement registers all Sokoban environments
# provided by this package
env_name = 'Sokoban-v0'
env = gym.make(env_name)

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))

for i_episode in range(1):#20
    observation = env.reset()

    for t in range(100):#100
        state = env.render(mode='one_hot')
        img = env.render(mode='rgb_array')
        # save this image as a file
        image = Image.fromarray(img)
        image.save('sokoban.png')
        action = env.action_space.sample()

        # Sleep makes the actions visible for users
        time.sleep(1)
        observation, reward, done, info = env.step(action)

        print(ACTION_LOOKUP[action], reward, done, info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.render()
            break

    env.close()

time.sleep(10)