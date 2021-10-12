from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import NTXentLoss, SelfSupTransform


class Component(nn.Module, ABC):
    def __init__(self, config, feature_extractor: nn.Module):
        super().__init__()
        self.config = config
        self.device = config['device'] if 'device' in config else 'cuda'
        self.feature = feature_extractor

        self.optimizer = NotImplemented
        self.lr_scheduler = NotImplemented

    @abstractmethod
    def forward(self, x):
        pass

    def setup_optimizer(self, optim_config):
        self.optimizer = getattr(torch.optim, optim_config['type'])(
            self.parameters(), **optim_config['options'])

    def setup_lr_scheduler(self, lr_config):
        self.lr_scheduler = getattr(torch.optim.lr_scheduler, lr_config['type'])(
            self.optimizer, **lr_config['options'])

    def _clip_grad_value(self, clip_value):
        for group in self.optimizer.param_groups:
            nn.utils.clip_grad_value_(group['params'], clip_value)

    def _clip_grad_norm(self, max_norm, norm_type=2):
        for group in self.optimizer.param_groups:
            nn.utils.clip_grad_norm_(group['params'], max_norm, norm_type)

    def clip_grad(self):
        clip_grad_config = self.config['clip_grad']
        if clip_grad_config['type'] == 'value':
            self._clip_grad_value(**clip_grad_config['options'])
        elif clip_grad_config['type'] == 'norm':
            self._clip_grad_norm(**clip_grad_config['options'])
        else:
            raise ValueError('Invalid clip_grad type: {}'
                             .format(clip_grad_config.type))

class SelfSup(Component):
    def __init__(self, config, feature_extractor, projection_in_dim, projection_out_dim):
        super().__init__(config, feature_extractor)
        self.projection = nn.Sequential(
            nn.Linear(projection_in_dim, projection_in_dim),
            nn.ReLU(),
            nn.Linear(projection_in_dim, projection_out_dim))

        self.transform = SelfSupTransform((config['x_h'], config['x_w'], config['x_c']))
        self.to(self.device)

    def init_ntxent(self, config, batch_size):
        self.ntxent_loss = NTXentLoss(self.device, batch_size,
                                      config['temperature'], config['use_cosine_similarity'])
    def forward(self, x):
        return self.projection(self.feature(x))

    def ntxent(self, zis, zjs):
        return self.ntxent_loss(zis, zjs)

    def get_selfsup_loss(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)

        f1 = self.forward(x1)
        f2 = self.forward(x2)

        # normalize projection feature vectors
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)

        loss = self.ntxent(f1, f2)
        return loss

class FineTune(Component):
    def __init__(self, config, feature_extractor, fc_in_dim):
        super().__init__(config, feature_extractor)
        self.fc = nn.Linear(fc_in_dim, config['nb_classes'])
        self.ce_loss = nn.NLLLoss()
        self.to(self.device)

    def forward(self, x):
        return self.fc(self.feature(x))

    def get_sup_loss(self, x, y):
        out = self.forward(x)
        out = F.log_softmax(out,dim=1)
        loss = self.ce_loss(out, y)
        return loss
