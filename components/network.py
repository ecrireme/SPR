from tensorboardX import SummaryWriter
from abc import ABC, abstractmethod
from components.component import SelfSup, FineTune
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18

class ResNetSelfSup(SelfSup):
    def __init__(self, config):
        resnet = resnet18(pretrained=False)
        projection_in_dim = resnet.fc.in_features
        projection_out_dim = config['projection_out_dim']
        resnet.fc = nn.Identity() # ignore fc layer
        super().__init__(config, resnet, projection_in_dim, projection_out_dim)

class ResNetFineTune(FineTune):
    def __init__(self, config):
        resnet = resnet18(pretrained=False)
        fc_in_dim = resnet.fc.in_features
        resnet.fc = nn.Identity() # ignore fc layer
        super().__init__(config, resnet, fc_in_dim)

class MLPSelfSup(SelfSup):
    def __init__(self, config):
        mlp = nn.Sequential(nn.Flatten(),
                            nn.Linear(config['x_h'] * config['x_w'], config['h1_dim']),
                            nn.ReLU(),
                            nn.Linear(config['h1_dim'], config['h2_dim']),
                            nn.ReLU())
        projection_in_dim = mlp[-2].out_features
        projection_out_dim = config['projection_out_dim']
        super().__init__(config, mlp, projection_in_dim, projection_out_dim)

class MLPFineTune(FineTune):
    def __init__(self, config):
        mlp = nn.Sequential(nn.Flatten(),
                            nn.Linear(config['x_h'] * config['x_w'], config['h1_dim']),
                            nn.ReLU(),
                            nn.Linear(config['h1_dim'], config['h2_dim']),
                            nn.ReLU())
        fc_in_dim = mlp[-2].out_features
        super().__init__(config, mlp, fc_in_dim)
