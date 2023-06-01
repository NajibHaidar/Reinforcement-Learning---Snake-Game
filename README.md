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

The implemenation began by studying the data given and it quickly became clear that a model function including cosine would make sense:

![image](https://user-images.githubusercontent.com/116219100/231102445-480b4510-7659-4146-a764-f623350de300.png)
*Figure 1: Plot of data points X and Y*

Once the model function was down and the number of parameters was determined, a conventional approach was taken to find the 'optimal' parameters that would yield minimum error through least-sqaures error. First, a helper function, **LSE** (least-squares error), which would calculate the least-squares error given 4 parameters as an array_like object, **c**, the given input data set, **x**, and the given output data set, **y**.

```
def LSEfit(c, x, y):
    E = np.sqrt(np.sum((c[0]*np.cos(c[1]*x)+c[2]*x+c[3]-y)**2)/n)
    return E
```

Then, optimization was applied using the SciPy library's optimize module which was imported as opt:

```
# set the initial guess for the parameters
c0 = np.array([3, 1*np.pi/4, 2/3, 32])

# perform optimization
res = opt.minimize(LSEfit, c0, args=(X, Y), method='Nelder-Mead')

# get the optimized parameters
c = res.x
```
As previously mentioned, the initial guess will vary the results since this is a nonlinear model with an unknown number of solutions. The initial guess used was the best I could find after trial and error. Optimization was done using the **LSEFit** function mentioned above and the Nelder-Mead method. I used this method because it is useful for optimizing functions that are not differentiable or whose derivatives are difficult to compute, however, do note that this method is a popular choice for optimization problems with a small number of variables, but can become inefficient in high-dimensional spaces or if the function being optimized is highly nonlinear or has multiple local minima.

After the optimized parameters were found, the minimum error (the furthest down the optimization algorithm could go) was found by passing these optimized parameters through the **LSEFit** function. The results will be discussed in **section IV.**.


In the next part of the project, the previously found parameters were studied more deeeply. Two of the four parameters were fixed at their optimal value while the other two were swept from 0 to 30 and a 2D loss (error) landscape was generated. Consider the case where A and B were the fixed parameters and C and D were swept across:

```
# FIX A B
# Initialize error grid
error_gridAB = np.zeros((len(C_range), len(D_range)))

# Loop through C and D ranges and compute error for each combination
for i, C in enumerate(C_range):
    for j, D in enumerate(D_range):
        # Compute error for fixed A and B and swept C and D
        error = compute_error(A_fixed, B_fixed, C, D, X, Y)
        # Store error in error grid
        error_gridAB[i, j] = error

# Generate x and y meshes from the ranges of C and D
C_mesh, D_mesh = np.meshgrid(C_range, D_range)

# Create a new figure and axis
fig, ax = plt.subplots()

# Plot the error grid as a pcolor map
pcm = ax.pcolormesh(C_mesh, D_mesh, error_gridAB, cmap='viridis', shading='auto')

# Add a colorbar to the plot
plt.colorbar(pcm).set_label('Error', fontweight='bold')
```

Note that color meshes are particularly useful in visualizing 2D arrays or grids (as can be seen in **section IV.**, hence why they were used. The error was stored in a error grid to be plotted against the meshes generated from the swept ranges. 

The above process was repeated for all six combinations of two fixed and two swept parameters.


Next, the first 20 data points were used as training data to determine the coeficients to fit a line, parabola and 19th degree polynomial to the data. Then, using these coeficients, the prediction accuracy of the model was tested against the last 10 data points using the least-squares error equation:

```
# Fit a parabola to the data
parabola_coeffs = np.polyfit(X[:20], Y[:20], deg=2)
parabola_predictions_train = np.polyval(parabola_coeffs, X[:20])
parabola_error_train = np.sqrt(np.sum((Y[:20] - parabola_predictions_train) ** 2)/20)

# Compute errors on test data
parabola_predictions_test = np.polyval(parabola_coeffs, X[-10:])
parabola_error_test = np.sqrt(np.sum((Y[-10:] - parabola_predictions_test) ** 2)/10)
```

Since all three of these fits are polynomials, the optimal coefficients were found using Numpy library's **polyfit** method. An initial guess was not required for this method since polynomials have a known number of solutions and therefore it iteratively minimizes the sum of squares of the residuals between the data and the polynomial fit until it determines the best-fit coefficients.

Finally, the same procedure was done except the training data became the first 10 and last 10 data points and then this model was fit to the 10 held out middle points (test data). 

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
