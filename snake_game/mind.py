## the mind will the rl code for driving the snake
## the states involves the measurement of the food and the walls and the body
## policy gradient will work for this 
## for body does it has to be the whole body or just the nearest segment ?, nearest would work 
## 11 state values , where is the danger located, straight, left or right, what is the direction of movement and where is the food located
## include only 3 actions , straight, left or right
## reset the game if wall or tail
## wall is 20 px
## change food after a time counter
## add init weights to model

import numpy as np
import torch as torch


class snake_mind:
    def __init__(self):
        self.model = self.build_model(12)
        self.action_space = np.arange(0,4)
        self.iterater = 0
        self.observation_arr = []
        self.action_arr = []
        self.reward_arr = []
        self.batch_size = 20
        self.gamma = 0.3
        self.learning_rate = 0.00008
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.part_number = 0
        self.boundary = 10
    
    def init_weights(self,m):
        if type(m) == torch.nn.Linear:
            m.weight.data.fill_(np.random.uniform(0.0,0.15))

    def build_model(self,first_layer):
        l1 = first_layer
        l2 = 720
        l3 = 320
        l4 = 100
        l5 = 4

        model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3, l4),
            torch.nn.ReLU(),
            torch.nn.Linear(l4, l5),
            torch.nn.Softmax(dim=-1)
            )

        # model.apply(self.init_weights)
        return model

    def nn_stuff(self, model, xdata, no_grad):
        if(no_grad):
            with torch.no_grad():
                qval = model(xdata)
        else:
            qval = model(xdata)
        
        return qval

    # include body measurements
    def observation(self, snake, food, body, direction,height,width):
        state = np.zeros(12)

        # up, down, left, right
        if snake[1] > food[1]:
            state[0] = 1
        if snake[1] < food[1]:
            state[1] = 1
        if snake[0] > food[0]:
            state[2] = 1
        if snake[0] < food[0]:
            state[3] = 1

        if direction == 'up':
            state[4] = 1
        elif direction == 'down':
            state[5] = 1
        elif direction == 'left':
            state[6] = 1
        elif direction == 'right':
            state[7] = 1

        ## get nearest node in the radius of 10 px
        part_vector = np.array(body)[2:]
        snake_vector = np.array(snake)
        distance_body = np.linalg.norm(part_vector - snake_vector,axis=1)
        closest_points = part_vector[distance_body <= 10]
        
        if snake[1] < self.boundary:
            state[8] = 1
        if snake[1] > width - self.boundary:
            state[9] = 1
        if snake[0] < self.boundary:
            state[10] = 1
        if snake[0] > height - self.boundary:
            state[11] = 1

        if len(closest_points) != 0:
            for pts in closest_points:
                if snake[1] > pts[1]:
                    state[8] = 1
                if snake[1] < pts[1]:
                    state[9] = 1
                if snake[0] > pts[0]:
                    state[10] = 1
                if snake[0] < pts[0]:
                    state[11] = 1

        return state
    

    def get_action(self,snake, food, body, direction,height,width):

        state = self.observation(snake, food, body, direction,height,width)

        state_tensor = torch.Tensor(state)
        action_prob = self.nn_stuff(self.model,state_tensor,False)
        action = np.random.choice(self.action_space, p = action_prob.data.numpy().flatten())

        
        self.observation_arr.append(state_tensor)
        self.action_arr.append(action)

        return action

    def save_model(self):
        torch.save(self.model.state_dict(), 'saved_model.pth')

    def get_reward(self, food, tail,food_distance,change, direction):
        self.reward = (1/food_distance)

        if food:
            self.reward = 10
        elif tail:
            self.reward = -5

        if change==0 and direction=='down':
            self.reward = self.reward - 0.5
        elif change == 1 and direction=='up':
            self.reward = self.reward - 0.5
        elif change == 2 and direction=='right':
            self.reward = self.reward - 0.5
        elif change == 3 and direction=='left':
            self.reward = self.reward - 0.5

        # print(self.reward)
        self.reward_arr.append(self.reward)


        if (len(self.observation_arr) > self.batch_size or food):
            
            rewards = np.full(len(self.reward_arr), self.gamma) ** np.arange(len(self.reward_arr)) * np.array(self.reward_arr)
            rewards = rewards[::-1].cumsum()[::-1]
            discounted_rewards = rewards - rewards.mean()

            # print(self.observation_arr)

            if food:
                self.observation_arr.pop()
                goal_state = np.zeros(12)
                self.observation_arr.append(torch.Tensor(goal_state))

            batch_observation = torch.stack(self.observation_arr)
            # batch_rewards.append(discounted_rewards) 
            batch_rewards = torch.Tensor(discounted_rewards).unsqueeze(1)
            # reward_transpose = torch.transpose(batch_rewards,0,1)

            # print("batch reward tensor is ",batch_rewards.size())
            # print("batch obs tensor is ",batch_observation)

            batch_action_prob = self.nn_stuff(self.model, batch_observation,False)
            action_tensor = torch.Tensor(self.action_arr).long().unsqueeze(1)

            # print("batch action prob ",batch_action_prob)
            # print("action tensor is ",action_tensor)

            logprob = torch.log(batch_action_prob)
            # print("logprob ",logprob)
            # print("gather ", torch.gather(logprob, 1, action_tensor))
            
            selected_prob = batch_rewards * torch.gather(logprob, 1, action_tensor)
            # print("selected prob ",selected_prob.size())

            loss = -selected_prob.sum()
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            self.observation_arr = []
            self.action_arr = []
            self.reward_arr = []