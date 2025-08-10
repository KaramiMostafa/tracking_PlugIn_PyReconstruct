from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def kl_gaussian(mu, logvar, prior_mu=0.0, prior_sigma=0.1):
    """
    KL divergence between N(mu, sigma) and N(prior_mu, prior_sigma) per weight element.
    """
    # log(sigma^2 / prior_sigma^2) + (sigma^2 + (mu-prior_mu)^2)/prior_sigma^2 - 1
    var = torch.exp(logvar)
    prior_var = prior_sigma**2
    t1 = logvar - math.log(prior_var)
    t2 = (var + (mu - prior_mu)**2) / prior_var
    return 0.5 * torch.sum(t1 + t2 - 1.0)

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.full((out_features,), -5.0))
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.bias_mu, 0.0)

    def forward(self, x, sample=True):
        if sample and self.training:
            w_std = torch.exp(0.5 * self.weight_logvar)
            b_std = torch.exp(0.5 * self.bias_logvar)
            w = self.weight_mu + w_std * torch.randn_like(w_std)
            b = self.bias_mu + b_std * torch.randn_like(b_std)
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)

    def kl(self):
        return (
            kl_gaussian(self.weight_mu, self.weight_logvar, self.prior_mu, self.prior_sigma)
            + kl_gaussian(self.bias_mu, self.bias_logvar, self.prior_mu, self.prior_sigma)
        )
