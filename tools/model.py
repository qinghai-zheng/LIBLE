import torch.nn as nn
import torch.nn.functional as F
import torch


class LIB_Encoder(nn.Module):
    def __init__(self, x_dim, h_dim=32):
        super(LIB_Encoder, self).__init__()
        self.h_dim = h_dim
        self.encode = nn.Sequential(
            nn.Linear(x_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 2 * self.h_dim))

    def forward(self, x):

        statistics = self.encode(x)
        mu = statistics[:,:self.h_dim]
        std = F.softplus(statistics[:,self.h_dim:])

        return mu, std

class LIB_Decoder(nn.Module):
    def __init__(self, h_dim,d_dim):
        super(LIB_Decoder, self).__init__()

        self.decode = nn.Sequential(
            nn.Linear(h_dim, d_dim))

    def forward(self, h):

        statistics = self.decode(h)

        return statistics


class LabelEnhanceNet(nn.Module):
    """LabelEnhanceNet is used to learn the label distribution d from the feature x"""
    def __init__(self, x_dim, d_dim):
        super(LabelEnhanceNet, self).__init__()
        self.coding = nn.Sequential(
            nn.Linear(x_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, d_dim))
        self.d_dim = d_dim

    def forward(self, x):
        d = self.coding(x)
        return d


class GapEstimationNet(nn.Module):
    """GapEstimationNet is used to estimation the gap level between the logical label l and the label distribution d"""
    def __init__(self, x_dim):
        super(GapEstimationNet, self).__init__()
        self.gap = nn.Sequential(
            nn.Linear(x_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1))

    def forward(self, x):
        sigma = self.gap(x)
        return sigma
