# SnakeGame-using-RL

Reinforcement Learning (RL) is a paradigm within the field of artificial intelligence that focuses on training agents to make sequential decisions by interacting with their environments. Unlike supervised learning, where models are trained on labeled datasets, RL relies on a reward-based system. Agents take actions in an environment, receive feedback in the form of rewards or penalties, and learn to optimize their behavior over time to maximize cumulative rewards. This trial-and-error approach enables RL to tackle complex problems where explicit solutions are difficult to specify. RL has found applications in various domains, such as robotics, game playing, finance, and natural language processing. It is a dynamic and evolving field that continually pushes the boundaries of what machines can learn and achieve through autonomous decision-making in real-world scenarios.

The classic game of snake is a perfect example of a MDP(Markov Decision Process) problem and hence can be solved using reinforcement learning methods.

The RL algorithms involve defining an environment, states, actions and rewards.
The state for our system involves a vector of size 12 which contains the state representation in the form 

## States
The state for our system involves a vector of size 12 which contains the state representation in the form-
### 1. Relative Position of the Food
The first 4 vectors define where the food is relative to the snake head based on the coordinates of the snake head and the food.
The value in the first 4 indexes is '1' based on if it is up, down , left or right. '0' otherwise.

### 2. Direction of Snake Movement
The next 4 index include the direction of movement of the snake head with respect to the game's frame, represented by '1' if true and '0' otherwise. Order of movement is Up, Down, Left and Right

### 3. Obstacle Information
Next 4 indexes includes the relative position of the obstacle with respect to the snake's head. An obstacle is considered only when the distance is below a specified threshold. The obstacles includes the wall and the nearest body part. The vector is represented by '1' is obstacle and '0' otherwise in the order Up, Down, Left and Right.

## Actions
Actions are represented by simple commands of Up, Down, Left and Right.

## Rewards
Reinforcement Learning works by maximizing the reward obtained in every iteration. For this the reward received by the agent is mapped as:
1. +10 is food
2. -5 is obstacle
3. $(1/\text{distance})$ otherwise where distance corresponds to the euclidean distance between the food and snake head. This step ensures possitive reward for any action leading towards the food

## Reinforcement Learning Method
Policy gradients represent a fundamental concept in reinforcement learning, serving as a powerful technique for training agents to navigate complex environments. Unlike traditional value-based methods that estimate the optimal action values, policy gradient methods directly learn the policy—the strategy an agent employs to make decisions. The core idea is to adjust the parameters of the policy in the direction that increases the likelihood of actions leading to higher rewards. Through the iterative process of interacting with the environment, collecting trajectories, and updating the policy based on the observed rewards, policy gradient methods have proven to efficient.

The central policy is estimated using a neural network which takes the state tensor as an input and gives the probabilty of each action. The action with the highest probability is used to drive the agent.

## Media
### Training
![video](https://github.com/saksham18kukreja/SnakeGame-using-RL/blob/main/snake_game/media/train_snake.gif)

The training process initialises the agent with a random policy to take actions and helps in exploring the environment and collect rewards.

### Trained
![video](https://github.com/saksham18kukreja/SnakeGame-using-RL/blob/main/snake_game/media/trained_snake.gif)

After succesful training and hyperparameter tuning the agent learns to reach the goal and avoid obstacles.

## Training Optimization
The model was trained on a smaller grid size to improve the likelihod of encountering food. The trained model on this small grid is generalized and can be scalled to larger grid size as can be seen in the trained video.

## Files
1. [Environment](https://github.com/saksham18kukreja/SnakeGame-using-RL/blob/main/snake_game/snake_game.py): This file contains the environment and game definition built using pygame.
2. [Policy](https://github.com/saksham18kukreja/SnakeGame-using-RL/blob/main/snake_game/mind.py): The file contains the code for policy gradient(RL) to train the agent.
3. [Test](https://github.com/saksham18kukreja/SnakeGame-using-RL/blob/main/snake_game/test.py): This is the code for testing and validating the trained policy on a larger grid size.


