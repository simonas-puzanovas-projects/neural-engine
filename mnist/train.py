from datasets import load_dataset
from neural_network import NeuralNetwork, LinearConfig, Activation
from engine import Value

BATCH_SIZE = 50

print("getting the data...")
ds = load_dataset("ylecun/mnist")

nn = NeuralNetwork(sequence=
    [LinearConfig(28*28,50, activation=Activation.RELU),
     LinearConfig(50,50, activation=Activation.RELU),
     LinearConfig(50,10,activation=Activation.LOG_SOFTMAX),
     ], learning_rate=0.01)

label_data = [ds["label"] for ds in ds["train"]]
image_data = [ds["image"].get_flattened_data() for ds in ds["train"]]

for i in range(0, len(label_data), BATCH_SIZE):
    print("batch:", i)

    batched_image_data = image_data[i:i+BATCH_SIZE]
    batched_label_data = label_data[i:i+BATCH_SIZE]

    valued_image_data = []
    valued_batch_data = []

    for image in batched_image_data:
        valued_image_data.append([Value(pixel/255) for pixel in image])
    for label in batched_label_data:
        valued_batch_data.append(Value(label))

    nn.train(valued_image_data, batched_label_data, classifier=True)