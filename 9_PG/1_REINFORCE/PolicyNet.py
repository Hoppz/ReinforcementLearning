import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, action_dim)
        # self.layer3 = nn.Linear(hidden_dim, output_dim)
        # 参数初始化
        nn.init.normal_(self.layer1.weight, mean=0, std=0.1)
        # nn.init.normal_(self.layer3.weight, mean=0, std=0.1)
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        x = self.layer2(x)
        # print(f'Output shape after layer2: {x.shape}')
        return F.softmax(x,dim=1)
