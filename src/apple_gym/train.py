import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from neural_network import NeuralNetwork, LinearConfig
from engine import Activation
from apple_gym import AppleGym
import pygame

apple_gym = AppleGym()

nn = NeuralNetwork(sequence=
    [LinearConfig(2,10, activation=Activation.RELU),
     LinearConfig(10,10, activation=Activation.RELU),
     LinearConfig(10,4,activation=Activation.LOG_SOFTMAX),
     ], learning_rate=0.005)

running = True
training_step = 0

solved = False

while running:
    training_step += 1

    if apple_gym.score == 150 and not solved:
        apple_gym.framerate = 10
        solved = True
        nn.save("models/test.json")

        print("Model saved, slowing down to 10fps.")


    state = apple_gym.state()
    prediction = nn.forward(state)

    highest_index = 0
    for i in range(len(prediction)):
        if prediction[highest_index].data < prediction[i].data:
            highest_index = i

    reward = apple_gym.tick(highest_index)

    loss = -prediction[highest_index] * reward

    loss.backward()
    nn.update_weights()
    nn.zero_grad()

    pygame.time.Clock().tick(apple_gym.framerate)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    training_step += 1






    






