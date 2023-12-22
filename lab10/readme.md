## Overview on Q-Learning

The core idea behind reinforcement learning is that agents learn by interacting with the environment, receiving feedback in the form of rewards or punishments based on their actions.

Q-learning is a popular technique in reinforcement learning that allows agents to learn optimal actions through trial and error. The algorithm employs a value function known as the Q-function, which represents the expected cumulative reward for taking a particular action in a given state. By iteratively updating the Q-function, Q-learning enables agents to make better decisions over time.

## Main classes

The class **TicTacToe** allows to play the game: we have methods that print the game field, check if a move is valid, check the winner and so on.

The class **TicTacToeRND** implements an agent playing randomly.

The class **TicTacToeLR** implements an agent adopting Q-learning.
The Q-function is represented using a Q-table, a two-dimensional table where rows correspond to states, and columns correspond to actions. The Q-value for a state-action pair (s, a) in the Q-table denotes the expected cumulative reward an agent can achieve by taking action ‘a’ in state ‘s’. Initially, the Q-table is populated with zero

## Training

The Q-learning algorithm follows the following iterative steps:
1. q_table is a dictionary containing the visited (states, actions) pairs. Not visisted pairs will have a 0 q-score.
2. Observe the current state of the environment.
3. Choose an action to take based on a trade-off between exploration and exploitation. This is done using an epsilon-greedy strategy, where the agent selects a random action with a certain probability and chooses the action with the highest Q-value with a complementary probability.
4. Perform the chosen action and observe the reward and the resulting next state.
5. Update the Q-value of the state-action pair using the Q-learning update rule:  
**Q(s, a) = (1 — α) * Q(s, a) + α * (R + γ * max(Q(s’, a’)))**  
**α** (alpha) is the learning rate, **γ** (gamma) is the discount factor that determines the importance of future rewards, **R** is the immediate reward obtained, and **max(Q(s’, a’))** represents the maximum Q-value for the next state.
6. Repeat steps 2–5 until predefined number of iterations (300_000).

The agent, while training, randomly starts first or second in order to learn playing in both situations.

## Scoring
The rewards are exponentially spaced: first (action, states) pairs are rewarded less with respect to the final ones wich are more decisive.  
For example if the agents wins in 4 moves the array of rewards are:  
[0.01, 0.16, 0.49, 1.]  
If the agent started first and the got a reward the scores are:  
[0.0001, 0.0016, 0.0049, 0.01]  
Wheras if it started second and got a draw:  
[0.0001, 0.0114, 0.0413, 0.09]  
I reward more a draw in the case the agent starts second since starting second is a disadventageous condition.

## Validation
In order to choose the best hyper-parameters values the validate function tried every possibile combination. Among all the best options the following configuraiton was chosen [.5, .7, .1] for learning rate, discount factor and exploration probability.

## Result
On 1000 matches the agent was able to win 93.1% of the times starting first and 91.4% of the times starting second