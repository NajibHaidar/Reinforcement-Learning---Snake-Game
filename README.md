# Reinforcement-Learning---Snake-Game

# MachineLearningNonLinearOptimization

### Table of Contents
[Abstract](#Abstract)
<a name="Abstract"/>

[Sec. I. Introduction and Overview](#sec-i-introduction-and-overview)     
<a name="sec-i-introduction-and-overview"/>

[Sec. II. Theoretical Background](#sec-ii-theoretical-background)     
<a name="sec-ii-theoretical-background"/>

[Sec. III. Algorithm Implementation and Development](#sec-iii-algorithm-implementation-and-development)
<a name="sec-iii-algorithm-implementation-and-development"/>

[Sec. IV. Computational Results](#sec-iv-computational-results)
<a name="sec-iv-computational-results"/>

[Sec. V. Summary and Conclusions](#sec-v-summary-and-conclusions)
<a name="sec-v-summary-and-conclusions"/>


### Abstract
This project explores the fascinating world of reinforcement learning through a classic game, Snake. We employ the Monte Carlo learning algorithm to train an AI agent (our snake) to play the game on a 4x4 grid. The main focus of the investigation is to understand the effects of the reward function and exploration rate (epsilon) on the agent's gameplay. Through an extensive series of experiments, models were trained with varying rewards for eating apples and dying, and exploration rates. The trained models were saved for analysis and future use, and their gameplay was observed to understand the impact of different parameters. This report presents the methodology, results, and insights derived from this investigation.

### Sec. I. Introduction and Overview
#### Introduction:

Reinforcement Learning (RL) is an area of machine learning that focuses on how an agent should take actions in an environment to maximize some notion of cumulative reward. The agent learns by continuously interacting with its environment, receiving feedback in the form of rewards or penalties. One of the fundamental challenges in RL is balancing exploration (trying out new, potentially better strategies) with exploitation (following the best strategy discovered so far).

#### Overview:

In this project, we delve into the world of RL through the lens of a classic game - Snake. The game offers a simple yet effective platform for investigating various aspects of RL, including the shaping of reward functions and the role of exploration.

Our agent, the snake, operates in a 4x4 grid, with the goal of eating as many apples as possible without colliding with the grid boundaries or its own body. The agent's actions at each step are determined by a policy, which is learned over time using the Monte Carlo learning algorithm.

The reward function plays a crucial role in RL as it guides the learning of the agent. We have defined four types of rewards in our game - for losing (colliding with the grid boundaries or the snake's own body), making an inefficient move (not eating an apple), making an efficient move (eating an apple), and for winning the game. The exploration rate, epsilon, determines the likelihood of the agent choosing a random action over the one suggested by its current policy. Both the reward function and the exploration rate were varied in our experiments to study their effects on the agent's gameplay.

Over the course of numerous episodes, the agent learns an optimal policy for playing the game, and the learned policy (Q-table) is saved for analysis and potential reuse. By observing the gameplay of the trained models and comparing their performance, we aim to uncover insights into the influence of different hyperparameters on the effectiveness of RL in game-playing scenarios.

This report will walk you through our methodology, present the results of our experiments, and offer insights into the intriguing dynamics between reward shaping, exploration, and game-playing performance in the context of reinforcement learning.

###  Sec. II. Theoretical Background

**Reinforcement Learning (RL)** forms the theoretical bedrock of this project. At its core, RL involves an agent learning to navigate an environment by taking certain actions, transitioning between states, and receiving rewards or penalties that inform its future actions. In the context of our project, the snake is the agent and the 4x4 grid constitutes the environment, with each cell in the grid representing a potential state.

**States and Actions**

In any given state, the agent has a set of possible actions it can take. In our game, the snake can choose to move in one of four directions - up, down, left, or right. Each action in a state leads to a transition to another state, and the agent receives a reward (positive or negative) based on this transition. The collection of states, actions, and the transition probabilities between them forms the Markov Decision Process (MDP) underlying our RL problem.

**Reward Structure and Policies**

The reward structure is a critical component of RL as it guides the learning process of the agent. We've set up four types of rewards in our game - for losing, making an inefficient move, making an efficient move, and winning. Based on the rewards it receives, the agent learns a policy, which is a mapping from states to actions. The policy dictates the action the agent should take when in a certain state.

**Monte Carlo Method**

The Monte Carlo (MC) method is a model-free reinforcement learning approach, meaning it does not require knowledge of the environment's dynamics, including transition probabilities. The MC method learns directly from episodes of experience. An episode is a sequence of states, actions, and rewards that goes from the beginning to the termination of an episode.

One of the key characteristics of the MC method is that it learns from complete episodes. This means that the method requires the episode to end before the value estimates can be updated. The agent can only learn from the final outcome of an episode, which makes the MC method well-suited for episodic tasks where all episodes terminate.

The updates in MC method are based on the returns following the first occurrence of each state-action pair within an episode. The return (denoted by G) is the cumulative discounted reward from a time-step t, given by the equation:

```
G_t = R_{t+1} + γR_{t+2} + γ^2R_{t+3} + ... = Σ_{k=0}^{∞} γ^kR_{t+k+1}
```

Here, R_{t+k+1} is the reward after k time-steps from time t, and γ is the discount factor. The return is basically the total future reward the agent expects to get from a certain state, considering the current policy.

**Q-Table and Q-Learning**

The Q-table is a simple yet powerful tool in reinforcement learning, especially for discrete state and action spaces. It is a table where each row represents a state, each column represents an action, and each cell represents the expected return (or Q-value) for a given state-action pair. The Q-value for a specific state-action pair is an estimate of the total reward that the agent expects to receive in the future, given it is in a specific state and takes a specific action, and thereafter follows a specific policy.

The Q-value is updated via the Q-learning algorithm, which is an off-policy method meaning it learns the value of the optimal policy irrespective of how the agent is exploring the environment. The update rule for Q-learning is:

```
Q(s,a) ← Q(s,a) + α [R + γ max_a' Q(s',a') - Q(s,a)]
```

Here, 
- `s` and `a` are the current state and action.
- `R` is the immediate reward the agent gets after taking action `a` in state `s`.
- `s'` is the new state after taking the action.
- `max_a' Q(s',a')` is the maximum Q-value over all possible actions in the new state `s'`. This represents the agent's estimate of the optimal future value.
- `γ` is the discount factor, which determines the importance of future rewards.
- `α` is the learning rate, which determines how much of the new information we acquire will overwrite the old information.

The Q-learning update rule uses the immediate reward and the estimate of the optimal future value to update the Q-value of the current state-action pair. This way, the agent gradually improves its Q-values, and consequently its policy, to align with the optimal policy that maximizes the expected return from each state.

**Discount Factor (Gamma)**

The discount factor, denoted by gamma, is a parameter that determines the importance of future rewards in the cumulative reward calculation. A gamma close to 0 makes the agent "short-sighted," caring only about immediate rewards, while a gamma close to 1 makes the agent "far-sighted," placing almost equal importance on all future rewards. In our experiments, we set gamma to 0.92, indicating a preference for future rewards but not entirely discounting immeadiate rewards.

**Exploration vs Exploitation**

Finally, an essential aspect of RL is the trade-off between exploration and exploitation, controlled by the exploration rate, epsilon. At the start of learning, the agent knows little about the environment, so it's beneficial to explore - take random actions - to gather information. As it learns more about the environment, it can start to exploit this knowledge, choosing the actions that it expects to yield the highest reward. A higher epsilon value encourages more exploration, while a lower value promotes exploitation.

This theoretical framework forms the backbone of our project, enabling us to investigate the impact of varying reward structures and exploration rates on the gameplay of our agent, the snake.

### Sec. III. Algorithm Implementation and Development

We begin with the **training.py** file which is the main script that drives the training and analysis of the model. It begins by importing the necessary modules and setting some initial hyperparameters. Here is a breakdown of the various sections:

The code snippet provided is the main script that drives the training and analysis of the model. It begins by importing the necessary modules and setting some initial hyperparameters. Here is a breakdown of the various sections:

**1. Import Statements and Initial Variables**

```python
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

grid_length = 4
n_episodes = 600000
gamma = 0.92 
num_games = 50
winning_score = 24
```
The script begins by importing necessary Python libraries and specific functions from the game and reinforcement learning (reinf) modules. The `grid_length`, `n_episodes`, `gamma`, `num_games`, and `winning_score` variables are then defined. These will be used as the core hyperparameters for the training of the snake game model.

**2. The `train_model` Function**

```python
def train_model(epsilon, rewards):
    # Training part
    env = SnakeEnv(grid_length=grid_length, with_rendering=False)
    q_table = perform_mc(env, n_episodes, epsilon, gamma, rewards)

    # Viz part
    env = SnakeEnv(grid_length=grid_length, with_rendering=False)
    total_reward, episode_rewards = show_games(env, num_games, q_table) 

    return q_table, total_reward, episode_rewards
```
This function performs the training and visualization of the game model. A `SnakeEnv` environment is created, and then the Monte Carlo simulation is performed using the `perform_mc` function. After training, the `show_games` function is called to visualize the game and return the total and episodic rewards.

**3. The `get_unique_filename` Function**

```python
def get_unique_filename(filename):
    base_name, extension = os.path.splitext(filename)
    counter = 1
    while os.path.exists(filename):
        filename = f"{base_name} ({counter}){extension}"
        counter += 1
    return filename
```
This function ensures that the filename used for saving the model is unique. It does this by appending a counter to the base filename if a file with the same name already exists.

**4. The `save_model` Function**

```python
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
    ...
    with open(full_path_pickle, 'wb') as f:
        pickle.dump(model_dict, f)
    ...
    with open(full_path_pretty, 'w') as f:
        pprint.pprint(model_dict, stream=f)
```
This function saves the trained model, including its hyperparameters and rewards, into a pickle file. It also saves a pretty-printed version of the model dictionary into a text file. The model's score is calculated as a percentage of the total reward over the product of `num_games` and `winning_score`.

**5. Hyperparameter Exploration**

```python
# Define range of parameters
epsilon_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
death_reward_range = [-3000, -1050, -100, -10, -1, 0, 10, 100, 2000, 10000, 100000000000] 
apple_reward_range = [100, 200, 500, 750, 1000, 2000, 3000, 5000, 10000, 20000, 100000000000]  

# Initialize lists to store results
epsilon_results = []
death_reward_results = []
apple_reward_results = []

# Vary epsilon, keeping rewards fixed
for epsilon in epsilon_range:
    q_table, total_reward, episode_rewards = train_model(epsilon, rewards=[-1050, -75, 2000, 10000000000000000])
    epsilon_results.append(total_reward / num_games)
    save_model(epsilon, [-1050, -75, 2000, 10000000000000000], q_table, total_reward, episode_rewards, 'epsilon')

# Vary death reward, keeping epsilon and apple reward fixed
for death_reward in death_reward_range:
    q_table, total_reward, episode_rewards = train_model(0.05, rewards=[death_reward, -75, 2000, 10000000000000000])
    death_reward_results.append(total_reward / num_games)
    save_model(0.05, [death_reward, -75, 2000, 10000000000000000], q_table, total_reward, episode_rewards, 'death')

# Vary apple reward, keeping epsilon and death reward fixed
for apple_reward in apple_reward_range:
    q_table, total_reward, episode_rewards = train_model(0.05, rewards=[-1050, -75, apple_reward, 10000000000000000])
    apple_reward_results.append(total_reward / num_games)
    save_model(0.05, [-1050, -75, apple_reward, 10000000000000000], q_table, total_reward, episode_rewards, 'apple')
```
In this section, the exploration of different hyperparameters is carried out. Three ranges are defined: one for the epsilon value (which controls the exploration vs exploitation trade-off), one for the reward received upon the snake's death, and one for the reward received for eating an apple. 

Then, three loops iterate over each range, one at a time, keeping the other two parameters constant. In each iteration, the `train_model` function is called to train the model, and the results (total reward and episode rewards) are saved. The `save_model` function is then used to save the model and its results for later analysis.

This methodical exploration of hyperparameters allows for the investigation of how each one affects the model's performance. This is an essential part of reinforcement learning, as it helps to fine-tune the model and understand the influence of each parameter on the gameplay.


Next, the `utils.py` script contains the main functionality for running the Monte Carlo algorithm and rendering the game, broken down into three main functions: `perform_mc`, `epsilon_greedy_policy`, and `show_games`.

1. `perform_mc` function: 


```python
def perform_mc(env, num_episodes, epsilon, gamma, rewards):
    """
    Perform monte carlo algorithm on num_episodes with epsilon moves on the
    Snake environment.

    Rewards should be given in the form of a list in the order :
    [Losing move, Inefficient move, Efficient move, Winning move]
    """
    action_space_size = 4
    q_table = defaultdict(lambda: np.zeros(action_space_size))
    state_action_count = defaultdict(lambda: np.zeros(action_space_size))
    for _ in tqdm(range(num_episodes)):
        episode_rewards = []
        episode_states = []
        episode_actions = []
        state, _ = env.reset()
        n_step = 0
        while True:
            actions = env.get_valid_actions(state)
            action = epsilon_greedy_policy(tuple(tuple(x) for x in state),
                                           actions,
                                           q_table,
                                           epsilon)
            next_state, reward, done, _ = env.step(action)
            # Reward is:
            # -1 if the move made the player lose
            # 0 if no apple is taken
            # 1 if apple is taken
            # 10 if the player won (screen full of snake)

            if reward == -1:
                reward = rewards[0]
            elif reward == 0:
                reward = rewards[1]
            elif reward == 1:
                reward = rewards[2]
            else:
                reward = rewards[3]
            episode_rewards.append(reward)
            episode_states.append(tuple(tuple(x) for x in state))
            episode_actions.append(action)
            if done or n_step > 100:
                break
            state = next_state
        unique_state_action_pairs = list(set(zip(episode_states, episode_actions)))

        for state, action in unique_state_action_pairs:
            indices = [i for i, (s, a) in enumerate(zip(episode_states, episode_actions)) if s == state and a == action]
            for i in indices:
                G = sum([episode_rewards[j]*gamma**(j-i) for j in range(i, len(episode_rewards))])
                state_action_count[state][action] += 1
                q_table[state][action] += (G - q_table[state][action]) / state_action_count[state][action]

    return q_table
```
This function is the heart of the Monte Carlo algorithm implementation. The function takes five parameters: the environment `env`, the number of episodes `num_episodes`, the exploration rate `epsilon`, the discount factor `gamma`, and the reward structure `rewards`.

In this function, the Q-table is initialized as a defaultdict, which is a dictionary-like object that provides all methods provided by a dictionary but takes a first argument (default factory) as a default data type for the dictionary. Here, the default data type is a numpy array of zeros of size equivalent to the action space size. The `state_action_count` is another defaultdict that tracks the number of times each action is taken for each state.

The function then starts iterating over the specified number of episodes. For each episode, it plays the game until it reaches a terminal state or a maximum number of steps is reached. The game state, action taken, and reward received are recorded for each step in the episode. These records are used to calculate the return (G), which is the sum of discounted rewards from each step. This return is used to update the Q-value for the state-action pair.

The Monte Carlo method is a First-Visit MC method, where the value of a state-action pair is updated based on the first visit to that pair in an episode. To implement this, a list of unique state-action pairs is created using the `set` function. Then, for each unique state-action pair, the indices of its occurrences in the episode are found. The return (G) is calculated for each occurrence, and the Q-value of the state-action pair is updated accordingly.

2. `epsilon_greedy_policy` function:

```python
def epsilon_greedy_policy(state, actions, q_table, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(actions)
    else:
        q_values = {action: q_table[state][action] for action in actions}
        return max(q_values, key=q_values.get)
```
This function implements the ε-greedy policy, which is a way of selecting actions such that the agent explores the environment and exploits its current knowledge. The function takes four parameters: the current state `state`, the available actions `actions`, the Q-table `q_table`, and the exploration rate `epsilon`.

The function selects an action based on the ε-greedy policy: with a probability of ε, it selects a random action (exploration), and with a probability of 1-ε, it selects the action with the highest Q-value for the current state (exploitation).

3. `show_games` function:

```python
def show_games(env, n_games, q_table, time_between_plays=0.0000000000001, max_time_per_game=30):  # Add max_time_per_game parameter
    # assert env.with_rendering, "You need to activate rendering on the \
    # environment to see the game"

    episode_rewards = []  # Store the reward for each game

    total_reward = 0
    for i in range(n_games):
        game_completed = False
        while not game_completed:
            start_time = time.time()  # Start the timer
            state, _ = env.reset()

            episode_reward = 0
            done = False
            while not done:
                # If the time limit has been exceeded, restart the game
                if time.time() - start_time > max_time_per_game:
                    break

                time.sleep(time_between_plays)
                actions = env.get_valid_actions(state)
                action = epsilon_greedy_policy(tuple(tuple(x) for x in state),
                                               actions,
                                               q_table,
                                               0.00)
                new_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = new_state
                
                # If the game is finished and the time limit was not exceeded, the game is completed
                if done:
                    game_completed = True
                    print(f"Reward on game {i} was {episode_reward}")

                    episode_rewards.append(episode_reward)  # Append the reward for this game
                    total_reward += episode_reward
                    
    return total_reward, episode_rewards  # Return the list of rewards
```

This function is used to render the game and watch the agent play after the Q-table has been learned. The function takes five parameters: the environment `env`, the number of games `n_games`, the Q-table `q_table`, the time delay between plays `time_between_plays`, and the maximum time for a game `max_time_per_game`.

In this function, the agent plays the specified number of games following the ε-greedy policy with ε set to 0 (i.e., always exploiting its current knowledge, no exploration). The total reward and the reward for each game are recorded. If a game exceeds the maximum time limit, it is restarted. The function returns the total reward and the list of rewards for each game.


### Sec. IV. Computational Results

In the first part, it can be seen from the plot below that the resulting fit was not very accurate but still pretty close to emulating the flucuations of the given data points. The **minimum error** was determined to be around **1.593** while the 4 optimal parameters found were **A=2.17**, **B=0.91**, **C=0.73**, **D=31.45**.

![image](https://user-images.githubusercontent.com/116219100/231102068-249d81e0-9f32-4f40-bfec-1a8ee8a93291.png)
*Figure 2: Fitting f(x) on X and Y using least-squares fit*


In the next part, the 2D loss (error) landscape's color maps yieled interesting results and provided insight to the affect certain parameters have on the function's number of minima.

![image](https://user-images.githubusercontent.com/116219100/231120449-630141c5-62f8-4375-89e4-be8d496d7aa1.png)
*Figure 3: Error Landscape, fix AB and sweep CD*

In figure 3 we can see that the effect of C when A and B are fixed is very minimal. For the entire range of C, the color is almost the same whereas as D ranges from 15 to 45 the error increases dramatically from well below 100 to over 500. This shows that the value of D has a very big impact when A and B are fixed.

![image](https://user-images.githubusercontent.com/116219100/231120490-b3197e4d-729a-43c4-a936-de2912acbc60.png)
*Figure 4: Error Landscape, fix AC and sweep BD*

Figure 4 is very interesting because we can see that when A and C are fixed, the error tends to minimize the closer B gets to around 17. Note that this is true for D being between 15 and 45. At first glance it looks like D does not have any influence, however, some ripples can be noticed at values of around 22, 27, 34, and 41 for D. We can see that at these values of D, the error is very minimal when B is between 10 and 15. 

![image](https://user-images.githubusercontent.com/116219100/231120534-47e8030b-c240-4e9d-92d9-63001816dc9a.png)
*Figure 5: Error Landscape, fix AD and sweep BC*

In figure 5 we can see that the effect of C when A and D are fixed is very minimal. For the entire range of C, the color is almost the same whereas as B ranges from 0 to 30 the error increases dramatically from well below 100 to over 500. This shows that the value of B has a very big impact when A and D are fixed.

![image](https://user-images.githubusercontent.com/116219100/231120588-bb073468-cb03-4b7b-badb-3af858a114d4.png)
*Figure 6: Error Landscape, fix BC and sweep AD*

Figure 6 is a good example of what a convex error surface would look like. Notice how no matter what point we start with here, if we follow the color descent gradient we will always end up in the minimum error region that is well below 5. We can also see that this lowest error occurs at almost A=17 and D=18 when B and C are fixed to their optimal values.

![image](https://user-images.githubusercontent.com/116219100/231120638-f7cfaf66-13c8-465e-8b23-35193e4023a6.png)
*Figure 7: Error Landscape, fix BD and sweep AC*

In figure 7 we can see that the effect of C when B and D are fixed is very minimal. For the entire range of C, the color is almost the same whereas as A ranges from 0 to 30 the error increases dramatically from well below 100 to over 500. This shows that the value of A has a very big impact when B and D are fixed.

![image](https://user-images.githubusercontent.com/116219100/231120679-7027a978-9c71-4eb3-8563-553a2a0a95e0.png)
*Figure 8: Error Landscape, fix BD and sweep AC*

Figure 8 is very interesting because we can see that when C and D are fixed, the error tends to minimize the closer B gets to around 0. Note that this is true for A being between 0 and 30. At first glance it looks like A does not have any influence, however, some ripples can be noticed at values of A. We can see that at these values of A, the error is very minimal, especially the closer B is to 0. It should also be noted that there are about 3 yellow lines showing that if A is at these values and B is closer to 30, the error increases by a good amount.


Moving on to testing different polynomial models against different sections of the data, we can see how training on different portions of the data leads to varying errors:

```
LINE MODEL, -Test End-:
Training Error: 2.242749386808539
Test Error: 3.4392356574390317

LINE MODEL, -Test Middle-:
Training Error: 1.8516699043293752
Test Error: 2.943490105614687


PARABOLA MODEL, -Test End-:
Training Error: 2.125539348277377
Test Error: 9.035130793088825

PARABOLA MODEL, -Test Middle-:
Training Error: 1.85083641159579
Test Error: 2.910426615782527


19th DEGREE POLYNOMIAL MODEL, -Test End-:
Training Error: 0.02835144302630829
Test Error: 30023572038.458946

19th DEGREE POLYNOMIAL MODEL, -Test Middle-:
Training Error: 0.16381508563760222
Test Error: 507.53804019224077
```

Recall that the given data points oscillate but still steadily increase as X increases.

For the line model, both types of tests yielded very low error in training and testing. However it seems that this model adapted better (although slim) to the version where we removed the middle values during training. This could be interpreted that since we are drawing a line here, the incline is what is important. The total incline can be better depicted by taking points in the beginning and then end of the entire data set and thus this would reduce error.

For the parabola model, the test error after training on the last 10 points was noticeably higher than that on the middle points. This could be interpreted that since the degree of curvature of a parabola will depend on future points, this was better captured when taking points in the beginning and then end of the entire data set and thus this would reduce error.

For the 19th degree polynomial model, the training error in both cases was almost nonexistent. This makes sense since a 19th degree polynomial can pass through all 10 data points with ease due to its degree. However, the test error was astronomical when testing the last 10 points and very large when testing the middle 10 points. This greatly descibes a phenomenon called overfitting. The model has been trained so strongly on the training data (by passing through each point) that when it is given data outside this data set its behaviour is very offset. The test on the middle points was definitely much better (although still not good) than the test on the end points. The interpretation for this is that since the problem here is overfitting, the gap presented by skipping the middle 10 points in training allows this model to be less overfit than its counterpart. The one tested on the first 20 points catches onto the given data points much more strongly and thus results in a much stronger effect of overfitting thus greatly increasing the error when the data changes from what the model was trained on.

### Sec. V. Summary and Conclusions

In this project, we explored the use of least-squares error and 2D loss landscapes for fitting a mathematical model to a dataset. We found that the Nelder-Mead method was effective in finding the optimal parameters for the model, and that the 2D loss landscape provided useful insights into the behavior of the model function and the sensitivity of the model to changes in the parameter values. We also found that the 19th degree polynomial was the best model for fitting this particular dataset, but that was only due to overfitting and caution should be exercised when extrapolating beyond the range of the data. Aditionally, we concluded how training data has a large effect on the flexibility and accuracy of a model. Overall, this project demonstrates the usefulness of data modeling and the power of mathematical models for analyzing and understanding complex datasets in machine learning.
