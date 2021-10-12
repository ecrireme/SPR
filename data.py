from abc import ABC, abstractmethod
from copy import deepcopy
import os
import random
import warnings
import colorful
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
from torch.utils.data.dataloader import default_collate
from tensorboardX import SummaryWriter
from PIL import Image, ImageFile

# =========
# Scheduler
# =========
class DataScheduler():
    def __init__(self, config):
        self.config = config
        self.schedule = config['data_schedule']
        self.datasets = {}
        self.eval_datasets = {}
        self.total_step = 0
        self.stage = -1

        # Prepare datasets
        for stage in self.schedule: # e.g, [['mnist', 0], ['mnist', 1]]
            for subset in stage['subsets']:
                dataset_name, _ = subset
                if dataset_name in self.datasets:
                    continue

                self.datasets[dataset_name] = DATASET[dataset_name](self.config)
                self.eval_datasets[dataset_name] = DATASET[dataset_name](self.config, train=False)
                self.total_step += len(self.datasets[dataset_name]) // self.config['batch_size']

        self.task_datasets = []
        for stage in self.schedule:
            subsets = []
            epoch = stage['epoch'] if 'epoch' in stage else 1
            for epoch in range(epoch):
                for dataset_name, subset_name in stage['subsets']:
                    subsets.append(self.datasets[dataset_name].subsets[subset_name])
            dataset = ConcatDataset(subsets)

            # data shuffling trick..
            # setting shuffle=True in below iter code works okay, but if you have experts trained
            # already with this trick, that would raise data inconsistency error
            random_indices = list(range(len(dataset)))
            random.shuffle(random_indices)
            dataset = Subset(dataset, random_indices)
            self.task_datasets.append(dataset)

    def __iter__(self):
        for t_i, task in enumerate(self.task_datasets):
            print(colorful.bold_green('\nProgressing to Task %d' % t_i).styled_string)
            collate_fn = task.dataset.datasets[0].dataset.collate_fn
            for data in DataLoader(task, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'],
                                   collate_fn=collate_fn, drop_last=True): # shuffle=True
                yield data, t_i

    def __len__(self):
        return self.total_step

    def eval(self, model, writer, step, eval_title):
        for eval_dataset in self.eval_datasets.values():
            eval_dataset.eval(model, writer, step, eval_title)

# ================
# Generic Datasets
# ================
class BaseDataset(Dataset, ABC):
    name = 'base'
    dataset_size = NotImplemented

    def __init__(self, config, train=True):
        self.config = config
        self.subsets = dict()
        self.train = train

    def __len__(self):
        return self.dataset_size

    def eval(self, model, writer: SummaryWriter, step, eval_title):
        if self.config['eval']:
            self._eval_model(model, writer, step, eval_title)

    @abstractmethod
    def _eval_model(self, model, writer: SummaryWriter, step, eval_title):
        raise NotImplementedError

    def collate_fn(self, batch):
        return default_collate(batch)


class ClassificationDataset(BaseDataset):
    num_classes = NotImplemented
    targets = NotImplemented

    def _eval_model(self, model, writer: SummaryWriter, step, eval_title):
        model = model.get_finetuned_model()

        totals = []
        corrects = []
        for subset_name, subset in self.subsets.items():
            data = DataLoader(subset, batch_size=self.config['eval_batch_size'],
                num_workers=self.config['eval_num_workers'], collate_fn=self.collate_fn)
            total = 0.
            correct = 0.
            for x, y in iter(data):
                with torch.no_grad():
                    x, y = x.to(model.device), y.to(model.device)
                    pred = model(x).argmax(dim=1)
                total += x.size(0)
                correct += (pred == y).float().sum()

            totals.append(total)
            corrects.append(correct)
            accuracy = correct / total
            writer.add_scalar('accuracy/%s/%s/%s' % (eval_title, self.name, subset_name),
                              accuracy, step)
            print('accuracy/%s/%s/%s' % (eval_title, self.name, subset_name), accuracy)

        # Overall accuracy
        total = sum(totals)
        correct = sum(corrects)
        accuracy = correct / total
        writer.add_scalar('accuracy/%s/%s/overall' % (eval_title, self.name),
                        accuracy, step)

        print('accuracy/%s/%s/overall' % (eval_title, self.name), accuracy)
        writer.flush()

class NoisyLabel(ClassificationDataset):
    org_targets = NotImplemented
    def add_symmetric_noise(self, source_class):
        for y in source_class:
            random_target = [t for t in source_class]
            random_target.remove(y)
            tindx = [i for i, x in enumerate(self.org_targets) if x == y]
            for i in tindx[:round(len(tindx)*self.config['corruption_percent'])]:
                self.targets[i] = random.choice(random_target)

    def add_asymmetric_noise(self, source_class, target_class):
        for s, t in zip(source_class, target_class):
            cls_idx = np.where(np.array(self.org_targets) == s)[0]
            n_noisy = int(self.config['corruption_percent'] * cls_idx.shape[0])
            noisy_sample_index = np.random.choice(list(cls_idx), n_noisy, replace=False)
            for idx in noisy_sample_index:
                self.targets[idx] = t

# =================
# Concrete Datasets
# =================
class MNIST(torchvision.datasets.MNIST, NoisyLabel):
    name = 'mnist'
    num_classes = 10

    def __init__(self, config, train=True):
        # Compose transformation
        transform_list = [transforms.Resize((config['x_h'], config['x_w'])),
                          transforms.ToTensor()]
        if config['x_c'] > 1:
            transform_list.append(lambda x: x.expand(config['x_c'], -1, -1))
        transform = transforms.Compose(transform_list)

        # Initialize super classes
        torchvision.datasets.MNIST.__init__(self, root=os.path.join(config['data_root'], 'mnist'),
                                            train=train, transform=transform, download=True)
        NoisyLabel.__init__(self, config, train)

        self.org_targets = deepcopy(self.targets)
        if train:
            if config['asymmetric_noise']:
                # 2->7, 3->8, 5<->6, 7->1
                source_class = [2, 3, 5, 6, 7]
                target_class = [7, 8, 6, 5, 1]
                self.add_asymmetric_noise(source_class, target_class)
            else:
                # symmetric noise.
                self.add_symmetric_noise(list(range(self.num_classes)))

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(self, torch.nonzero((self.targets == y)).squeeze(1).tolist())
        self.dataset_size = len(self.targets)

    def __getitem__(self, idx):
        x, y = torchvision.datasets.MNIST.__getitem__(self, idx)
        if not self.train:
            return x, y
        return x, y, y != self.org_targets[idx], idx

class CIFAR10(torchvision.datasets.CIFAR10, NoisyLabel):
    name = 'cifar10'
    num_classes = 10

    def __init__(self, config, train=True):
        transform = transforms.Compose([transforms.Resize((config['x_h'], config['x_w'])),
                                        transforms.ToTensor()])
        torchvision.datasets.CIFAR10.__init__(self, root=os.path.join(config['data_root'], 'cifar10'),
                                              train=train, transform=transform, download=True)
        NoisyLabel.__init__(self, config, train)

        self.org_targets = deepcopy(self.targets)
        if train:
            if config['asymmetric_noise']:
                # bird->airplane, cat<->dog, deer->horse, truck->automobile
                source_class = [9, 2, 3, 5, 4]
                target_class = [1, 0, 5, 3, 7]
                self.add_asymmetric_noise(source_class, target_class)
            else:
                # symmetric noise
                self.add_symmetric_noise(list(range(self.num_classes)))

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(self, torch.nonzero((torch.Tensor(self.targets) == y)).squeeze(1).tolist())
        self.dataset_size = len(self.targets)

    def __getitem__(self, idx):
        x, y = torchvision.datasets.CIFAR10.__getitem__(self, idx)
        if not self.train:
            return x, y

        return x, y, y != self.org_targets[idx], idx


class CIFAR100(torchvision.datasets.CIFAR100, NoisyLabel):
    name = 'cifar100'
    num_classes = 100

    def __init__(self, config, train=True):
        transform = transforms.Compose([transforms.Resize((config['x_h'], config['x_w'])),
                                        transforms.ToTensor()])
        torchvision.datasets.CIFAR100.__init__(self, root=os.path.join(config['data_root'], 'cifar100'),
                                               train=train, transform=transform, download=True)
        NoisyLabel.__init__(self, config, train)

        self.org_targets = deepcopy(self.targets)
        if train:
            if config['superclass_noise']:
                # symmetric noise within superclass
                super_classes = [["beaver", "dolphin", "otter", "seal", "whale"],
                                 ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                                 ["orchid", "poppy", "rose", "sunflower", "tulip"],
                                 ["bottle", "bowl", "can", "cup", "plate"],
                                 ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
                                 ["clock", "keyboard", "lamp", "telephone", "television"],
                                 ["bed", "chair", "couch", "table", "wardrobe"],
                                 ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                                 ["bear", "leopard", "lion", "tiger", "wolf"],
                                 ["bridge", "castle", "house", "road", "skyscraper"],
                                 ["cloud", "forest", "mountain", "plain", "sea"],
                                 ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                                 ["fox", "porcupine", "possum", "raccoon", "skunk"],
                                 ["crab", "lobster", "snail", "spider", "worm"],
                                 ["baby", "boy", "girl", "man", "woman"],
                                 ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                                 ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                                 ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                                 ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                                 ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],]
                for super_cls in super_classes:
                    cls_idx = [self.class_to_idx[c] for c in super_cls]
                    self.add_symmetric_noise(cls_idx)
            else:
                self.add_symmetric_noise(list(range(self.num_classes)))

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(self, torch.nonzero((torch.Tensor(self.targets) == y)).squeeze(1).tolist())
        self.dataset_size = len(self.targets)

    def __getitem__(self, idx):
        x, y = torchvision.datasets.CIFAR100.__getitem__(self, idx)
        if not self.train:
            return x, y
        return x, y, y != self.org_targets[idx], idx


class WEBVISION(NoisyLabel):
    name = 'webvision'
    num_classes = 14

    def __init__(self, config, train=True):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        transform_list = [transforms.Resize((config['x_h'], config['x_w'])),
                          lambda x: x.convert('RGB'),
                          transforms.ToTensor(),
                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        self.transform = transforms.Compose(transform_list)
        NoisyLabel.__init__(self, config, train)

        self.idx_to_realname = list()
        with open(os.path.join(config['data_root'], self.name, 'info', 'synsets.txt'), 'r') as f:
            for i, cls in enumerate(f.readlines()):
                self.idx_to_realname.append(cls)

        self.data = list()
        self.targets = list()

        if train:
            infos_pth = os.path.join(config['data_root'], self.name, 'info', 'train_filelist_google.txt')
        else:
            infos_pth = os.path.join(config['data_root'], self.name, 'info', 'val_filelist.txt')

        LABEL_LIST = [412, 480, 506, 395, 421, 121, 498, 762, 48, 896, 32, 414, 147, 436]

        self.data = list()
        self.targets = list()

        with open(infos_pth, 'r') as f:
            for info in f.readlines():
                name, label = info.split(' ')
                if not train:
                    name = os.path.join('val_images', name)

                label = int(label)
                if label not in LABEL_LIST:
                    continue
                self.data.append(name)
                self.targets.append(LABEL_LIST.index(label))

        self.targets = torch.LongTensor(self.targets)
        self.org_targets = self.targets

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(self, torch.nonzero(self.targets == y).squeeze(1).tolist())
        self.dataset_size = len(self.targets)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x = self.transform(Image.open(os.path.join(self.config['data_root'], self.name, self.data[idx])))
        y = self.targets[idx]
        if not self.train:
            return x, y
        return x, y, True, idx # in webvision, we don't know which data is corrupted.


DATASET = {
    MNIST.name: MNIST,
    CIFAR10.name: CIFAR10,
    CIFAR100.name: CIFAR100,
    WEBVISION.name: WEBVISION,
}
