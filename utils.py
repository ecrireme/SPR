import logging
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import kornia
import scipy.stats as stats
from colorlog import ColoredFormatter

def setup_logger():
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red',
        }
    )

    logger = logging.getLogger('example')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger

def override_config(config, override):
    # Override options
    for option in override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                raise ValueError('{} is not defined in config file. '
                                 'Failed to override.'.format(address))
            here = here[key]
        if keys[-1] not in here:
            raise ValueError('{} is not defined in config file. '
                             'Failed to override.'.format(address))
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)
    return config

class SelfSupTransform():
    def __init__(self, image_shape):
        transform = [
            kornia.augmentation.RandomResizedCrop(size=image_shape[:2]),
            kornia.augmentation.RandomHorizontalFlip()]
        if image_shape[2] == 3:
            transform.append(kornia.augmentation.ColorJitter(0.5, 0.5, 0.5, 0.2, p=0.5))
        self.transform = transforms.Compose(transform)
    def __call__(self, image):
        return self.transform(image)


class BetaMixture1D(object):
    """This code is based on the https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/utils.py"""
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    @staticmethod
    def fit_beta_weighted(x, w):
        def weighted_mean(x, w):
            return np.sum(w * x) / np.sum(w)

        x_bar = weighted_mean(x, w)
        s2 = weighted_mean((x - x_bar)**2, w)
        alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
        beta = alpha * (1 - x_bar) /x_bar
        return alpha, beta

    @staticmethod
    def outlier_remove(x):
        # outliers detection
        max_perc = np.percentile(x, 95)
        min_perc = np.percentile(x, 5)
        x = x[(x<=max_perc) & (x>=min_perc)]
        x_max = max_perc
        x_min = min_perc + 10e-6
        return x, x_min, x_max

    @staticmethod
    def normalize(x, x_min, x_max):
        # normalized the centrality for bmm
        x = (x - x_min) / (x_max - x_min + 1e-6)
        x[x >= 1] =  1 -10e-4
        x[x <= 0] = 10e-4
        return x

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        wl = self.weighted_likelihood(x, y)
        p = self.probability(x)
        pos = wl / (p + self.eps_nan)

        wl_inf = np.isinf(wl)
        p_inf = np.isinf(p)

        # inf / inf -> 1
        pos[wl_inf & p_inf] = 1.
        return pos

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unstable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = self.fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = self.fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x):
        x = np.array((self.lookup_resolution * x).astype(int))
        x[x < 0] = 0
        x[x == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x]

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


class NTXentLoss(nn.Module):
    """This code is based on the https://github.com/chagmgang/simclr_pytorch/blob/master/nt_xent_loss.py"""
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)

        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
