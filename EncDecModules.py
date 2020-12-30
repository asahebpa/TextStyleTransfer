import torch.nn as nn
import torch.nn.functional as F


class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.fc1 = nn.Linear(768, 600)
        self.fc2 = nn.Linear(600, 500)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        self.fc3 = nn.Linear(500, 600)
        self.fc4 = nn.Linear(600, 768)
    def forward(self, x):
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
