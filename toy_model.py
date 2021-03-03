from torch import nn
import torch.functional as F


class energy_nn(nn.Module):
    def __init__(self, input_size, hidden1_size=128, hidden2_size=128):
        super(energy_nn, self).__init__()
        self.dense1 = nn.Linear(input_size, hidden1_size)
        self.active1 = nn.ReLU()
        self.dense2 = nn.Linear(hidden1_size, hidden2_size)
        self.active2 = nn.ReLU()
        self.dense_out = nn.Linear(hidden2_size, 1)

    def forward(self, input):
        x = self.active1(self.dense1(input))
        x = self.active2(self.dense2(x))
        x = self.dense_out(x)
        return x


class force_nn(nn.Module):
    def __init__(self, input_size, hidden1_size=128, hidden2_size=128):
        super(force_nn, self).__init__()
        self.dense1 = nn.Linear(input_size, hidden1_size)
        self.active1 = nn.ReLU()
        self.dense2 = nn.Linear(hidden1_size, hidden2_size)
        self.active2 = nn.ReLU()
        self.dense_out = nn.Linear(hidden2_size, input_size)

    def forward(self, input):
        x = self.active1(self.dense1(input))
        x = self.active2(self.dense2(x))
        x = self.dense_out(x)
        return x

class modular_nn(nn.Module):
    def __init__(self, input_module, module, repeat, output_module):
        super(modular_nn, self).__init__()
        modules = [input_module] + [module for i in range(repeat)] + [output_module]
        self.net = nn.Sequential(*modules)

    def forward(self, input):
        x = self.net(input)
        return x
