import pygame
from engine import Value
from enum import Enum
import random
import math

MAP_COLOR = (150, 200, 150)
PLAYER_COLOR = (50, 50, 200)
APPLE_COLOR = (50, 200, 50)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

MAP_SIZE = 10
REC_SIZE = SCREEN_WIDTH / MAP_SIZE
REC_GAP = 2.0



class AppleGym:
    def __init__(self):
        self.player_position = [5, 5]
        self.apple_position = [random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)]
        self.player_steps = 0
        self.max_steps = 20
        self.score = 0
        self.framerate = 240
        self.max_score = 0

        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    def reset(self):
        self.player_position = [random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)]
        self.apple_position = [random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)]
        self.player_steps = 0
        self.score = 0

    def state(self):
        apple_x = Value((self.apple_position[0] - self.player_position[0]))
        apple_y = Value((self.apple_position[1] - self.player_position[1]))

        return [apple_x, apple_y]

    def tick(self, input):

        last_distance = math.dist(self.player_position, self.apple_position)

        if input == 0:
            self.player_position[0] -= 1
        elif input == 1:
            self.player_position[0] += 1
        elif input == 2:
            self.player_position[1] -= 1
        elif input == 3:
            self.player_position[1] += 1

        new_distance = math.dist(self.player_position, self.apple_position)

        reward = 0

        if new_distance > last_distance:
            reward = -10
        else: reward += 1

        if self.player_position[0] > MAP_SIZE-1 or self.player_position[0] < 0:
            reward = -1
            self.reset()

        elif self.player_position[1] > MAP_SIZE-1 or self.player_position[1] < 0:
            reward = -1
            self.reset()

        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
                pygame.draw.rect(self.screen, MAP_COLOR, pygame.Rect(REC_SIZE*i+REC_GAP/2.0, REC_SIZE*j+REC_GAP/2.0, REC_SIZE-REC_GAP, REC_SIZE-REC_GAP))

        pygame.draw.rect(self.screen, PLAYER_COLOR, pygame.Rect(REC_SIZE*self.player_position[0]+REC_GAP/2.0, REC_SIZE*self.player_position[1]+REC_GAP/2.0, REC_SIZE-REC_GAP, REC_SIZE-REC_GAP))
        pygame.draw.rect(self.screen, APPLE_COLOR, pygame.Rect(REC_SIZE*self.apple_position[0]+REC_GAP/2.0, REC_SIZE*self.apple_position[1]+REC_GAP/2.0, REC_SIZE-REC_GAP, REC_SIZE-REC_GAP))
        pygame.display.flip()

        if self.player_position == self.apple_position:
            self.score += 1.0
            reward = self.score
            self.player_steps = 0
            self.apple_position = [random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)]


            if self.score > self.max_score:
                self.max_score = self.score

            print("score: ", self.score)

        if self.player_steps >= self.max_steps:
            reward = -1
            self.reset()
        
        self.player_steps += 1
        return Value(reward)

