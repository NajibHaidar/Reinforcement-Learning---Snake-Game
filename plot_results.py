import os
import pickle
import matplotlib as plt

def get_rewards_and_hyperparams(directory, hyperparam, index):
    """
    Iterate through the files in a given directory, extract the total_reward
    along with a specified hyperparameter, and then store them as pairs for
    plotting.

    Parameters:
    directory (str): The directory to search through.
    hyperparam (str): The name of the hyperparameter to extract.

    Returns:
    list of tuple: A list of (total_reward, hyperparam_value) pairs.
    """
    results = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):  # Only process .pkl files
            full_path = os.path.join(directory, filename)

            # Open the file and load the model_dict
            with open(full_path, 'rb') as f:
                model_dict = pickle.load(f)

            # Extract the total_reward and the specified hyperparameter
            total_reward = model_dict['rewards_returned']['total_reward']
            if (hyperparam == 'rewards_set'):
                hyperparam_value = model_dict['hyperparameters'][hyperparam][index]
            else:
                hyperparam_value = model_dict['hyperparameters'][hyperparam]

            # Add the (total_reward, hyperparam_value) pair to the results
            results.append((total_reward, hyperparam_value))

    return results

epsilon_results = get_rewards_and_hyperparams('epsilon_pkl', 'epsilon', -1)
death_reward_results = get_rewards_and_hyperparams('death_pkl', 'rewards_set', 0)
apple_reward_results = get_rewards_and_hyperparams('apple_pkl', 'rewards_set', 2)



import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit

# # Separate the results into two lists for plotting
# rewards, epsilon_values = zip(*epsilon_results)

# # Create the scatter plot
# plt.scatter(rewards, epsilon_values)

# # Add labels to the axes and a title to the plot
# plt.xlabel('Total Reward')
# plt.ylabel('Epsilon')
# plt.title('Epsilon vs. Total Reward')

# # Display the plot
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Separate the results into two lists for plotting
# rewards, death_values = zip(*death_reward_results)

# # Create the scatter plot
# plt.scatter(rewards, death_values)

# # Set the y-axis to a symlog scale
# plt.yscale('symlog')

# # Add labels to the axes and a title to the plot
# plt.xlabel('Total Reward')
# plt.ylabel('Death Reward Values (Symlog Scale)')
# plt.title('Death Reward Values vs. Total Reward')

# # Display the plot
# plt.show()



# Separate the results into two lists for plotting
rewards, apple_values = zip(*apple_reward_results)

# Create the scatter plot
plt.scatter(rewards, apple_values)

# Set the y-axis to a symlog scale
plt.yscale('symlog')

# Add labels to the axes and a title to the plot
plt.xlabel('Total Reward')
plt.ylabel('Apple Reward Values (Symlog Scale)')
plt.title('Apple Reward Values vs. Total Reward')

# Display the plot
plt.show()






# # Plotting results for varying death reward
# plt.figure(figsize=(10, 6))
# plt.plot(death_reward_results, 'o-')
# plt.title("Performance vs Death Reward")
# plt.xlabel("Death Reward")
# plt.ylabel("Average Score")
# plt.grid()
# plt.show()

# # Plotting results for varying apple reward
# plt.figure(figsize=(10, 6))
# plt.plot(apple_reward_results, 'o-')
# plt.title("Performance vs Apple Reward")
# plt.xlabel("Apple Reward")
# plt.ylabel("Average Score")
# plt.grid()
# plt.show()