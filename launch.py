from game.Snake import Snake
from reinf.SnakeEnv import SnakeEnv
from reinf.utils import perform_mc, show_games

import pickle
from collections import defaultdict
import numpy as np

# Winning everytime hyperparameters
grid_length = 4
n_episodes = 500000
epsilon = 0.01 # exploration parameter, probability to explore
gamma = 0.92 # discount factor, how much you want ot care about the future, (if set to 0 then we just care abt the next move only [gamma + 1] moves ahead)
rewards = [-1050, -75, 2000, 10000000000000000] # [Losing move, inefficient move, efficient move, winning move]

# Playing part
# game = Snake((800, 800), grid_length)
# game.start_interactive_game()

# Training part
# env = SnakeEnv(grid_length=grid_length, with_rendering=False)
# q_table = perform_mc(env, n_episodes, epsilon, gamma, rewards)

# Load from pickle file
with open('Initial Models/0.97%_model.pkl', 'rb') as f:
    model_dict = pickle.load(f)

# Convert q_table back to defaultdict
q_table = defaultdict(lambda: np.zeros(4), model_dict['q_table'])

# Viz part
env = SnakeEnv(grid_length=grid_length, with_rendering=False)
num_games = 100
total_reward, episode_rewards = show_games(env, num_games, q_table)

