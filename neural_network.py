from engine import Value, Layer, loss, Activation

class LinearConfig:
    def __init__(self, inputs, outputs, activation=None):
        self.inputs = inputs
        self.outputs = outputs
        self.activation = activation

class NeuralNetwork:
    def __init__(self, sequence, learning_rate = 0.005):
        self.sequence = []
        self.learning_rate = learning_rate

        for i in sequence:
            if isinstance(i, LinearConfig):
                self.sequence.append(Layer(i.inputs, i.outputs, i.activation))

    def zero_grad(self):
        for l in self.sequence:
            for p in l.parameters():
                p.grad = 0.0

    def update_weights(self):
            for l in self.sequence:
                for p in l.parameters():
                    p.data -= self.learning_rate * p.grad

    def train(self, input_matrix, target_matrix, learning_rate):
        training_step = 0

        for i in range(len(input_matrix)):
            input_output_buffer = input_matrix[i]

            for seq in self.sequence:
                input_output_buffer = seq.forward(input_output_buffer)

            loss_result = loss(target_matrix[i], input_output_buffer)
            loss_result.backward()

            print("step:", training_step, "loss:", round(loss_result.data, 5))

            self.update_weights(learning_rate)
            self.zero_grad()

            training_step += 1


    def forward(self, input_matrix):
        input_output_buffer = input_matrix

        for seq in self.sequence:
            input_output_buffer = seq.forward(input_output_buffer)

        return input_output_buffer
