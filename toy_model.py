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
