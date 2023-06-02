from ctypes import alignment
import pickle
import pprint
import os

from game.Snake import Snake
from reinf.SnakeEnv import SnakeEnv
from reinf.utils import perform_mc, show_games

import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

# Winning everytime hyperparameters
grid_length = 4
n_episodes = 600000
gamma = 0.92 # discount factor, how much you want ot care about the future, (if set to 0 then we just care abt the next move only [gamma + 1] moves ahead)
num_games = 50

def train_model(epsilon, rewards):
    # Training part
    env = SnakeEnv(grid_length=grid_length, with_rendering=False)
    q_table = perform_mc(env, n_episodes, epsilon, gamma, rewards)

    # Viz part
    env = SnakeEnv(grid_length=grid_length, with_rendering=False)
    total_reward, episode_rewards = show_games(env, num_games, q_table) 

    return q_table, total_reward, episode_rewards

winning_score = 24

def get_unique_filename(directory, filename):
    base_name, extension = os.path.splitext(filename)
    counter = 1
    full_path = os.path.join(directory, filename)  # Join the directory with the filename
    while os.path.exists(full_path):
        filename = f"{base_name} ({counter}){extension}"
        full_path = os.path.join(directory, filename)  # Update the full path with the new filename
        counter += 1
    return filename

def save_model(epsilon, rewards, q_table, total_reward, episode_rewards, dir_name):
    # Convert defaultdict to regular dict before pickling
    model_dict = {
        'hyperparameters': {
            'num_episodes': n_episodes,
            'epsilon': epsilon,
            'gamma': gamma,
            'rewards_set': rewards
        },
        'rewards_returned': {
            'episode_rewards': episode_rewards,
            'total_reward': total_reward
        },
        'q_table': dict(q_table),
    }

    model_score = model_dict['rewards_returned']['total_reward'] / (num_games * winning_score)
    model_score = model_score * 100

    # Define the directory and create it if it doesn't exist

    dir_name_pickle = f'{dir_name}_pkl'
    dir_name_pretty = f'{dir_name}_pretty'

    os.makedirs(dir_name_pickle, exist_ok=True)
    os.makedirs(dir_name_pretty, exist_ok=True)

    # Save the model
    filename_pickle = f'{model_score:.0f}%_model.pkl'
    filename_pickle = get_unique_filename(dir_name_pickle, filename_pickle)

    # Combine the directory name and the filename
    full_path_pickle = os.path.join(dir_name_pickle, filename_pickle)

    with open(full_path_pickle, 'wb') as f:
        pickle.dump(model_dict, f)

    # Save pretty-printed string in a file
    filename_pretty= f'{model_score:.0f}%_pretty.txt'
    filename_pretty = get_unique_filename(dir_name_pretty, filename_pretty)

    full_path_pretty = os.path.join(dir_name_pretty, filename_pretty)

    with open(full_path_pretty, 'w') as f:
        pprint.pprint(model_dict, stream=f)

# Define range of parameters
epsilon_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
death_reward_range = [-1] 
apple_reward_range = [200, 500, 1000, 5000, 10000, 20000, 100000000000]  

# Initialize lists to store results
epsilon_results = []
death_reward_results = []
apple_reward_results = []

# Vary epsilon, keeping rewards fixed
# for epsilon in epsilon_range:
    # print(f'Training epsilon: {epsilon}')
#     q_table, total_reward, episode_rewards = train_model(epsilon, rewards=[-1050, -75, 2000, 10000000000000000])
#     epsilon_results.append(total_reward / num_games)
#     save_model(epsilon, [-1050, -75, 2000, 10000000000000000], q_table, total_reward, episode_rewards, 'epsilon')

# Vary death reward, keeping epsilon and apple reward fixed
for death_reward in death_reward_range:
    print(f'Training death reward: {death_reward}')
    q_table, total_reward, episode_rewards = train_model(0.05, rewards=[death_reward, -75, 2000, 10000000000000000])
    death_reward_results.append(total_reward / num_games)
    save_model(0.05, [death_reward, -75, 2000, 10000000000000000], q_table, total_reward, episode_rewards, 'death')

# Vary apple reward, keeping epsilon and death reward fixed
for apple_reward in apple_reward_range:
    print(f'Training apple reward: {apple_reward}')
    q_table, total_reward, episode_rewards = train_model(0.05, rewards=[-1050, -75, apple_reward, 10000000000000000])
    apple_reward_results.append(total_reward / num_games)
    save_model(0.05, [-1050, -75, apple_reward, 10000000000000000], q_table, total_reward, episode_rewards, 'apple')


# # Saving results
# results = {
#     'epsilon_range': epsilon_range,
#     'epsilon_results': epsilon_results,
#     'death_reward_range': death_reward_range,
#     'death_reward_results': death_reward_results,
#     'apple_reward_range': apple_reward_range,
#     'apple_reward_results': apple_reward_results
# }

# with open('results.pkl', 'wb') as f:
#     pickle.dump(results, f)

# with open('results_pretty.txt', 'w') as f:
#     pprint.pprint(results, stream=f)

# # Now, we can plot epsilon_results, death_reward_results, and apple_reward_results
# # against epsilon_values, death_reward_values, and apple_reward_values respectively

# # Plotting results for varying epsilon
# plt.figure(figsize=(10, 6))
# plt.plot(epsilon_range, epsilon_results, 'o-')
# plt.title("Performance vs Epsilon")
# plt.xlabel("Epsilon")
# plt.ylabel("Average Score")
# plt.grid()
# plt.show()

# # Plotting results for varying death reward
# plt.figure(figsize=(10, 6))
# plt.plot(death_reward_range, death_reward_results, 'o-')
# plt.title("Performance vs Death Reward")
# plt.xlabel("Death Reward")
# plt.ylabel("Average Score")
# plt.grid()
# plt.show()

# # Plotting results for varying apple reward
# plt.figure(figsize=(10, 6))
# plt.plot(apple_reward_range, apple_reward_results, 'o-')
# plt.title("Performance vs Apple Reward")
# plt.xlabel("Apple Reward")
# plt.ylabel("Average Score")
# plt.grid()
# plt.show()

# print(get_unique_filename('death_pkl', '100%_model.pkl'))