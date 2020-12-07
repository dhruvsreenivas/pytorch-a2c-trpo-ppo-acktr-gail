"""
Modified standard Pytorch distributions used for sampling actions given states in the policy.
KL-divergence methods for each distribution is also included.
"""
from abc import ABC

import torch
import torch.nn as nn
from a2c_trpo_ppo_acktr.utils import AddBias, init


class FixedCategorical(torch.distributions.Categorical, ABC):
    def sample(self, **kwargs):
        return super().sample(kwargs).unsqueeze(-1)

    def log_prob(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


@torch.distributions.register_kl(FixedCategorical, FixedCategorical)
def calculate_kl(p: FixedCategorical, q: FixedCategorical):
    return torch.distributions.kl._kl_categorical_categorical(p, q)


class FixedNormal(torch.distributions.Normal, ABC):
    def log_prob(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


@torch.distributions.register_kl(FixedNormal, FixedNormal)
def calculate_kl(p: FixedCategorical, q: FixedCategorical):
    return torch.distributions.kl._kl_normal_normal(p, q)


class FixedBernoulli(torch.distributions.Bernoulli, ABC):
    def log_prob(self, actions):
        return super().log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()

@torch.distributions.register_kl(FixedBernoulli, FixedBernoulli)
def calculate_kl(p: FixedCategorical, q: FixedCategorical):
    return torch.distributions.kl._kl_bernoulli_bernoulli(p, q)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

        self.dist = None

    def forward(self, x):
        x = self.linear(x)
        self.dist = FixedCategorical(logits=x)
        return self.dist


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))
        self.dist = None

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros_like(action_mean.size())

        action_logstd = self.logstd(zeros)
        self.dist = FixedBernoulli(action_mean, action_logstd.exp())
        return self.dist


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))
        self.dist = None

    def forward(self, x):
        x = self.linear(x)
        self.dist = FixedBernoulli(logits=x)
        return self.dist