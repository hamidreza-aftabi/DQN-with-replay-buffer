import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, state_size, action_size):        
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        y = torch.relu(self.fc1(x))
        y = torch.relu(self.fc2(y))
        y = self.fc3(y)
        return y
        raise NotImplementedError()

    def select_action(self, state):
        self.eval()
        z = self.forward(state)
        self.train()
        return z.max(1)[1].view(1, 1).to(torch.long)
