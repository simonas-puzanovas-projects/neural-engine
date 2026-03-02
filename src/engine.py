import random
import math
from enum import Enum

class Activation(Enum):
    LOG_SOFTMAX = 0
    RELU = 1

class Value:
    def __init__(self, value):
        self.data = value
        self.grad = 0.0
        self._backward = lambda: None
        self.children = set()

    
    def __mul__(self, other):
        out = Value(self.data * other.data)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        out.children.add(self)
        out.children.add(other)
        return out


    def __add__(self, other):
        out = Value(self.data + other.data)

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        out.children.add(self)
        out.children.add(other)
        return out

    
    def __sub__(self, other):
        out = Value(self.data - other.data)
        
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
            
        out._backward = _backward
        out.children.add(self)
        out.children.add(other)
        return out
    
    def __pow__(self, other):
        out = Value(self.data ** other)

        def _backward():
            self.grad += (other * self.data**(other-1))*out.grad

        out._backward = _backward
        out.children.add(self)
        return out
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __neg__(self):
        out = Value(-self.data)

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        out.children.add(self)
        return  out

    def exp(self):
        out = Value(math.exp(self.data))

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        out.children.add(self)
        return out
    
    def relu(self):
        out = Value(self.data if self.data > 0 else 0)

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0) * out.grad

        out._backward = _backward
        out.children.add(self)
        return out

    
    
    def log(self):
        out = Value(math.log(self.data))

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        out.children.add(self)
        return out

    def backward(self):
        topo = []
        visited = set()
        self.grad = 1.0

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for i in reversed(topo):
            i._backward()

class Layer:
    def __init__(self, in_size, out_size, activation=None):
        self.weights = []
        self.biases = []
        self.in_size = in_size
        self.out_size = out_size
        self.last_input = None
        self.activation = activation

        for i in range(in_size):
            self.weights.append([])
            for _ in range(out_size):
                self.weights[i].append(Value(random.uniform(-1.0, 1.0)))

        for i in range(out_size):
            self.biases.append(Value(random.random()))
    
    def forward(self, input_matrix):
        if self.in_size != len(input_matrix):
            raise ValueError("wrong input size")

        self.last_input = input_matrix

        neurons = [Value(0) for _ in range(self.out_size)]

        for i in range(self.out_size):
            for j in range(self.in_size):
                neurons[i] += input_matrix[j] * self.weights[j][i]
            neurons[i] += self.biases[i]

        if self.activation == Activation.LOG_SOFTMAX:
            neurons = log_softmax(neurons)

        elif self.activation == Activation.RELU:
            neurons = [n.relu() for n in neurons]

        return neurons

    def parameters(self):
        return [w for row in self.weights for w in row] + self.biases
    
def loss(y_true, y_pred):
    total_loss = Value(0)    

    for yt, yp in zip(y_true, y_pred):
        total_loss += (yt - yp) ** 2

    return (total_loss * Value(1.0/len(y_pred)))

def max(value_array):
    out = value_array[0]

    for value in value_array[1:]:
        if value.data > out.data:
            out = value

    return out

def log_softmax(value_array):
    max_val = max(value_array)

    shifted = [v - max_val for v in value_array]
    exps = [v.exp() for v in shifted]

    sum = Value(0)
    for e in exps:
        sum += e

    log_sum_exp = sum.log()

    return [v - log_sum_exp for v in shifted]
