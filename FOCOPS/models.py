import numpy as np
import torch.nn as nn
import torch
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available(), torch.backends.cudnn.enabled)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation='tanh', log_std=-0.5):
        super().__init__()

        hidden_dim = 128

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.l1 = nn.Linear(obs_dim, hidden_dim)

        self.l2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.l3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_layer = nn.Linear(hidden_dim, act_dim)

        self.logstd_layer = nn.Parameter(torch.ones(1, act_dim) * log_std)

        self.mean_layer.weight.data.mul_(0.1)
        self.mean_layer.bias.data.mul_(0.0)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, obs, hidden):

        a = torch.tanh(self.l1(obs))

        self.l2.flatten_parameters()
        a, hidden = self.l2(a, hidden)

        a = torch.tanh(self.l3(a))

        mean = self.mean_layer(a)

        if len(mean.size()) == 1:
            mean = mean.view(1, -1)

        logstd = self.logstd_layer.expand_as(mean)
        std = torch.exp(logstd)

        return mean, logstd, std, hidden

    def get_initial_states(self):

        h_0 = torch.zeros((
            self.l2.num_layers,
            1,
            self.l2.hidden_size),
            dtype=torch.float)
        h_0 = h_0.to(device=device)

        c_0 = torch.zeros((
            self.l2.num_layers,
            1,
            self.l2.hidden_size),
            dtype=torch.float)
        c_0 = c_0.to(device=device)

        return (h_0, c_0)
    def get_act(self, obs, hidden, deterministic = False):
        obs = torch.tensor(
            obs.reshape(1, -1)).to(device)[:, None, :]

        mean, _, std, hidden = self.forward(obs, hidden)
        if deterministic:
            return mean
        else:
            return torch.normal(mean, std), hidden

    def logprob(self, obs, act, hidden):
        mean, _, std, hidden = self.forward(obs, hidden)
        normal = Normal(mean, std)
        return normal.log_prob(act).sum(-1, keepdim=True), mean, std, hidden



class Value(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64, 64), activation='tanh'):
        super().__init__()

        hidden_dim = 128

        self.obs_dim = obs_dim

        self.l1 = nn.Linear(obs_dim, hidden_dim)

        self.l2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.l3 = nn.Linear(hidden_dim, hidden_dim)

        self.v_head = nn.Linear(hidden_dim, 1)

        self.v_head.weight.data.mul_(0.1)
        self.v_head.bias.data.mul_(0.0)

        self.device = torch.device("cuda")

    def forward(self, obs, hidden, batch=False):
        # if not batch:
        #     obs = torch.tensor(obs.reshape(1, -1)).to(device)[None, :]
        # else:
        obs = torch.cat([obs], -1)
        q1 = torch.tanh(self.l1(obs))
        self.l2.flatten_parameters()

        q1, hidden = self.l2(q1, hidden)
        q1 = torch.tanh(self.l3(q1))
        q1 = self.v_head(q1)
        return q1


