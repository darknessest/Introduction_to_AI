from math import exp
from random import seed
from random import random


# Initialize a network
def initialize_network(n_inputs, n_outputs, n_hidden):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[mapper[j]] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, alpha):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += alpha * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += alpha * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, target, alpha, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            # target = [0 for i in range(n_outputs)]
            target = {x: 0 for x in target}
            # last value in the row is the one to be predicted (int)
            target[row[-1]] = 1

            sum_error += sum([(target[mapper[i]] - outputs[i]) ** 2 for i in range(len(target))])
            backward_propagate_error(network, target)
            update_weights(network, row, alpha)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, alpha, sum_error))


def load_dataset(inputs_file, targets_file):
    dataset = []
    with open(inputs_file) as fi:
        for line in fi:
            str_floats = line.split()
            list_floats = list(map(float, str_floats))
            dataset.append(list_floats)
    fi.close()
    with open(targets_file) as ft:
        i = 0
        expected = dict()
        for line in ft:
            dataset[i].append(int(line))
            expected[int(line)] = 0
            i += 1
    ft.close()

    return dataset, expected


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Test training backpropagation algorithm
seed(1)

# load dataset:
X, Y = load_dataset("input_data", "target_data")
mapper = list(Y)

n_inputs = len(X[0]) - 1
n_outputs = len(set([row[-1] for row in X]))

# network = initialize_network(n_inputs, n_outputs, n_hidden=2)
train_network(network, X, Y, 0.075, 1000, n_outputs)
for layer in network:
    print(layer)
