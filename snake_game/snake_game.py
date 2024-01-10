#!usr/bin/python3

import numpy as np
import pygame
from pygame.locals import *
import random
import copy
from mind import snake_mind
from test import snake_mind_test
import time
import os


pygame.font.init()
class food:
    def __init__(self,height,width):
        self.boundary_food = 25
        self.food_x = random.randint(self.boundary_food,width-self.boundary_food)
        self.food_y = random.randint(self.boundary_food,height-self.boundary_food)


class game_window:

    #game window parameters
    def __init__(self):
        self.game_window_height = 480 #480 #100   
        self.game_window_width = 720 #720 #150
        self.background_color = (255,255,255)
        self.screen = pygame.display.set_mode((self.game_window_width,self.game_window_height))
        self.font = pygame.font.SysFont("Comic Sans MS", 25)
        self.boundary_reach = False
        self.body_counter = 0
        self.random_food = food(self.game_window_height,self.game_window_width)
        self.food_error = 10
        self.tail_error = 5
        self.moving_direction = "right"
        self.tail_touched = False
        self.fps_controller = pygame.time.Clock()
        self.eat_food = False
        # self.mind = snake_mind()
        self.mind_test = snake_mind_test()
        self.test = 1
        self.play_game = 1
        


    def create_rectangle(self,top_left_x,top_left_y,width,height,color):
        rect = Rect(top_left_x,top_left_y,width,height)
        pygame.draw.rect(self.screen,color,rect)


    def main_game_loop(self):

        running = True
        change = "0"
        snake = [[100,100],[90,100],[80,100]]
        snake_head = [100,100]
        simulation = True
        food_counter_init = time.time()

        while (simulation):
            
            self.body_counter = 0
            self.boundary_snake = 10
            snake_head = [np.random.randint(self.boundary_snake,self.game_window_width-self.boundary_snake), np.random.randint(self.boundary_snake,self.game_window_height - self.boundary_snake)]
            snake = [[snake_head[0],snake_head[1]],[snake_head[0]-10,snake_head[1]],[snake_head[0]-10,snake_head[1]]]
            self.tail_touched = False
            
            while (running):
                
                if ((time.time() - food_counter_init) > 10):
                    food_counter_init = time.time()
                    self.random_food = food(self.game_window_height,self.game_window_width)

                self.eat_food = False
                snake_body = copy.deepcopy(snake)

                self.screen.fill(self.background_color)

                if not self.play_game:                
                    if self.test:
                        change = self.mind_test.get_action(snake_head,[self.random_food.food_x, self.random_food.food_y],snake,self.moving_direction,self.game_window_height,self.game_window_width)
                    else:
                        change = self.mind.get_action(snake_head,[self.random_food.food_x, self.random_food.food_y],snake,self.moving_direction,self.game_window_height,self.game_window_width)
                    
                else:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False

                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_LEFT:
                                change = 2

                            if event.key == pygame.K_RIGHT:
                                change = 3

                            if event.key == pygame.K_UP:
                                change = 0

                            if event.key == pygame.K_DOWN:
                                change = 1

            
                # print(change, self.moving_direction)
                if change == 0 and self.moving_direction != "down":
                    self.moving_direction = "up"
                elif change == 1 and self.moving_direction != "up":
                    self.moving_direction = "down"
                elif change == 3 and self.moving_direction != "left":
                    self.moving_direction = "right"
                elif change == 2 and self.moving_direction != "right":
                    self.moving_direction = "left"

                # self.moving_direction = self.temp_moving_direction

                if self.moving_direction == "right":
                    snake_head[0] += 5
            
                if self.moving_direction == "left":
                    snake_head[0] -= 5 
                
                if self.moving_direction == "up":
                    snake_head[1] -= 5
                
                if self.moving_direction == "down":
                    snake_head[1] += 5
            
                
                if(snake_head[0]>self.game_window_width or snake_head[0]<0 or snake_head[1]>self.game_window_height or snake_head[1] <0):
                    self.tail_touched = True

                for i in range(1,len(snake)):
                    snake[i] = copy.deepcopy(snake_body[i-1])
                    
                snake[0][0] = snake_head[0]
                snake[0][1] = snake_head[1]


                for body in snake:
                    self.create_rectangle(body[0],body[1],10,10,'red')
                

                self.create_rectangle(self.random_food.food_x,self.random_food.food_y,10,10,"black")
                
                distance_food =  np.sqrt((snake_head[0] - self.random_food.food_x)**2 + (snake_head[1] - self.random_food.food_y)**2)
    
                if(distance_food < self.food_error):
                    snake.insert(len(snake),snake[-1])
                    self.random_food = food(self.game_window_height,self.game_window_width)
                    self.eat_food = True
                    food_counter_init = time.time()
                    self.body_counter += 1
                
                for i in range(1,len(snake)):
                    distance_body = np.sqrt((snake_head[0] - snake[i][0])**2 + (snake_head[1] - snake[i][1])**2)
                    if (distance_body < self.tail_error):
                        self.tail_touched = True

                if not self.test:
                    self.mind.get_reward(self.eat_food, self.tail_touched, distance_food,change,self.moving_direction)

                text_surface = self.font.render("%d" %tuple([self.body_counter]),False,(0,0,0))
                self.screen.blit(text_surface,(self.game_window_width-4*self.boundary_snake,self.boundary_snake))
                pygame.display.set_caption("Snake Game")
                pygame.display.flip()

                self.fps_controller.tick(100)

                if self.tail_touched:
                    break
        
        if not self.test and not self.play_game:
            self.mind.save_model()
        
        print("game over")
        print(self.body_counter)
        pygame.quit()


if __name__ == "__main__":
    game = game_window()
    game.main_game_loop()