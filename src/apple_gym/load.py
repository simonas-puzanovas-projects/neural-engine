import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from neural_network import NeuralNetwork
from apple_gym import AppleGym
import pygame

apple_gym = AppleGym()
apple_gym.framerate = 20.0

nn = NeuralNetwork(load="models/test.json")

running = True

while running:

    state = apple_gym.state()
    prediction = nn.forward(state)

    highest_index = 0
    for i in range(len(prediction)):
        if prediction[highest_index].data < prediction[i].data:
            highest_index = i

    apple_gym.tick(highest_index)
    pygame.time.Clock().tick(apple_gym.framerate)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()







    






