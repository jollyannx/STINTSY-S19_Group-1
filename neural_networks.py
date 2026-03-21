import numpy as np
import torch
import torch.nn as nn
import torch.nn.init


ACTIVATION_TYPES = (nn.ReLU, nn.Sigmoid, nn.Tanh)

class DataLoader(object):
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(self.X.shape[0])
        np.random.seed(1)

    def shuffle(self):
        np.random.shuffle(self.indices)

    def get_batch(self, mode='train'):
        X_batch, y_batch = [], []
        if mode == 'train':
            self.shuffle()
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            X_batch.append(torch.tensor(self.X[batch_indices], dtype=torch.float32))
            y_batch.append(torch.tensor(self.y[batch_indices], dtype=torch.float32))

        return X_batch, y_batch

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, list_hidden, activation='sigmoid'):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.list_hidden = list_hidden
        self.activation = activation
        self.dropout = nn.Dropout(p=0.2)
        self.create_network()
        self.init_weights()

    def create_network(self):
        layers = []
        # First Layer
        layers.append(nn.Linear(self.input_size, self.list_hidden[0]))
        layers.append(self.get_activation(self.activation))

        # Hidden Layers
        for i in range(len(self.list_hidden) - 1):
            layers.append(nn.Linear(self.list_hidden[i], self.list_hidden[i+1]))
            layers.append(self.get_activation(self.activation))

        # Output Layer
        layers.append(nn.Linear(self.list_hidden[-1], self.num_classes))
        self.layers = nn.Sequential(*layers)

    def init_weights(self):
        torch.manual_seed(2)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                nn.init.constant_(module.bias, 0)

    def get_activation(self, mode='sigmoid'):
        if mode == 'tanh': return nn.Tanh()
        elif mode == 'relu': return nn.ReLU(inplace=True)
        return nn.Sigmoid()

    def forward_manual(self, x, verbose=False):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = torch.matmul(x, layer.weight.t()) + layer.bias
            else:
                x = layer(x)
                x = self.dropout(x)
                if verbose:
                    print(f"Output shape: {x.shape}")

        logits = x
        probabilities = torch.softmax(logits, dim=1)
        return logits, probabilities

    def forward(self, x, verbose=False):
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, ACTIVATION_TYPES):
                x = self.dropout(x)
                if verbose:
                    print(f"Output shape: {x.shape}")

        logits = x
        probabilities = torch.softmax(logits, dim=1)

        return logits, probabilities

    def predict(self, probabilities):
        return torch.argmax(probabilities, dim=1)
