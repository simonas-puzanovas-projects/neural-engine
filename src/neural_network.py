import os
from engine import Value, Layer, loss, Activation
import json

class LinearConfig:
    def __init__(self, inputs, outputs, activation=None):
        self.inputs = inputs
        self.outputs = outputs
        self.activation = activation

class NeuralNetwork:
    def __init__(self, sequence=None, learning_rate = 0.005, load=None):
        self.sequence = []
        self.learning_rate = learning_rate

        if load != None:
            with open(load, "r") as file:

                converted_json = json.load(file)
                for linear in converted_json["sequence"]:
                    layer = Layer(0, 0, None)
                    layer.weights = [[Value(w) for w in w_array] for w_array in linear["weights"]]
                    layer.biases = [Value(b) for b in linear["biases"]]
                    layer.activation = Activation(linear["activation"])
                    layer.in_size = len(linear["weights"])
                    layer.out_size = len(linear["weights"][0])

                    self.sequence.append(layer)


        elif sequence != None:
            for i in sequence:
                if isinstance(i, LinearConfig):
                    self.sequence.append(Layer(i.inputs, i.outputs, i.activation))

    def save(self, dir):
        data = {"sequence": []}

        for i in range(len(self.sequence)):
            data["sequence"].append({
                "activation": self.sequence[i].activation.value,
                "weights": [[w.data for w in w_array] for w_array in self.sequence[i].weights],
                "biases": [b.data for b in self.sequence[i].biases]
            })

        os.makedirs(os.path.dirname(dir), exist_ok=True)
        with open(dir, "w") as file:
            json.dump(data, file)

    def zero_grad(self):
        for l in self.sequence:
            for p in l.parameters():
                p.grad = 0.0

    def update_weights(self):
            for l in self.sequence:
                for p in l.parameters():
                    p.data -= self.learning_rate * p.grad

    def train(self, input_matrix, target_matrix, classifier=False):
        training_step = 0

        for i in range(len(input_matrix)):
            input_output_buffer = input_matrix[i]

            for seq in self.sequence:
                input_output_buffer = seq.forward(input_output_buffer)

            if classifier:
                highest_output_index = 0
                for j in range(1,len(input_output_buffer)):
                    if input_output_buffer[j].data > input_output_buffer[highest_output_index].data:
                        highest_output_index = j

                loss = -input_output_buffer[target_matrix[i]]
                print("step:", training_step, "loss:", round(loss.data, 5),"correct prediction: ", highest_output_index == target_matrix[i])

                loss.backward()
                self.update_weights()
                self.zero_grad()

            else:
                loss_result = loss(target_matrix[i], input_output_buffer)
                print("step:", training_step, "loss:", round(loss_result.data, 5))

                loss_result.backward()
                self.update_weights()
                self.zero_grad()

            training_step += 1


    def forward(self, input_matrix):
        input_output_buffer = input_matrix

        for seq in self.sequence:
            input_output_buffer = seq.forward(input_output_buffer)

        return input_output_buffer
