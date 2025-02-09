import torch
import torchvision
from torchvision import ops
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

from utils import *

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        model = vgg16(weights = VGG16_Weights)
        required_layers = list(model.children())[0][:17]
        self.backbone = nn.Sequential(*required_layers)
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)

#dev2
