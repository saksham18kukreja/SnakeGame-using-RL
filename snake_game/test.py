
import numpy as np
import torch as torch


class snake_mind_test:
    def __init__(self):
        self.model = self.build_model(12)
        self.model.load_state_dict(torch.load('saved_model.pth'))
        self.model.eval()
        self.action_space = np.arange(0,4)
        self.iterater = 0
        self.observation_arr = []
        self.action_arr = []
        self.reward_arr = []
        self.batch_size = 15
        self.gamma = 0.3
        self.learning_rate = 0.0001
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
        action_prob = self.nn_stuff(self.model,state_tensor,True)
        
        action = np.argmax(action_prob.numpy().flatten())

        return action

  