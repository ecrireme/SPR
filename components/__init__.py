from typing import Dict
from .component import Component
from .network import ResNetSelfSup, ResNetFineTune, MLPSelfSup, MLPFineTune

Net: Dict[str, Component] = {
    'resnet_simclr18': ResNetSelfSup,
    'resnet_simclr18_ft': ResNetFineTune,
    'mlp': MLPSelfSup,
    'mlp_ft': MLPFineTune
}
