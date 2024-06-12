from torch import nn
import torch.nn.functional as F


class ValueNet(nn.Module):

    def __init__(self, state_dim, hidden_dim ):
        super(ValueNet, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

        nn.init.normal_(self.layer1.weight, 0, 0.1)
        nn.init.normal_(self.layer2.weight, 0, 0.1)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        return self.layer2(x)
