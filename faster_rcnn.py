import os

from utils import *
from model import *

import torch
import torchvision
from torchvision import ops
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import matplotlib.patches as patches


# Create dataset and dataloader
annotations_path = "VOCdevkit/VOC2007/Annotations"
img_size = (200, 200)
img_dir = "VOCdevkit/VOC2007/JPEGImages"

dataset = ObjectDetectionDataset(annotations_path, img_dir, img_size)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Create decode class name
idx2name = decode_class(dataset.name2idx)
idx2name.update({-1: ''})

# trainning function
def training(model, dataloader, learning_rate, n_epoch, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    loss_list = []
    for i in range(n_epoch):
        loss_o = 0
        pbar = tqdm(dataloader)
        for img_data, gt_boxes, gt_classes in pbar:

            img_data = img_data.to(device)
            gt_boxes = gt_boxes.to(device)
            gt_classes = gt_classes.to(device)
            model.to(device)
            loss = model(img_data, gt_boxes, gt_classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_o += loss.item()
            loss_list.append(loss_o)
            pbar.set_description(f'Epoch {i+1}/{n_epoch} Loss: {loss.item():.4f}')
    return loss_list

# Feature Extractor
feature_extractor =  FeatureExtractor()

# Proposal Module
proposal_module = ProposalModule()

# Two Stage Detector
two_stage_detector = TwoStageDetector(feature_extractor, proposal_module, len(idx2name))

# training
learning_rate = 1e-3
n_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
loss_list = training(two_stage_detector, dataloader, learning_rate, n_epochs, device)


