import xml.etree.ElementTree as ET
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import torchvision
import torch
from torchvision import ops
import torch.nn.functional as F
import torch.optim as optim

def parse_annotation(annotation_path, imgs_dir):
    #traverse xml
    gt_class_all = []
    gt_boxes_all = []
    imgs_list_dir = []
    for xml_file in os.listdir(annotation_path):
        file_path = os.path.join(annotation_path, xml_file)
        tree = ET.parse(file_path)
        root = tree.getroot()
        all_object = root[7:] + root[-1:]
        img_dir = os.path.join(imgs_dir, root[1].text)
        imgs_list_dir.append(img_dir)
        list_gt_class = []
        list_gt_boxes = []
        for object in all_object:
            list_gt_class.append(object[0].text)
            bndbox = object.find("bndbox")
            list_gt_boxes.append([float(bndbox[0].text), float(bndbox[1].text),
                                  float(bndbox[2].text), float(bndbox[3].text)])
        gt_boxes_all.append(torch.Tensor(list_gt_boxes))
        gt_class_all.append(list_gt_class)
    return imgs_list_dir, gt_class_all, gt_boxes_all

def encode_class(gt_classes_all):
    result = dict()
    list_all_class = [item for list_item in gt_classes_all for item in list_item]
    set_all_class = set(list_all_class)
    classes = list(set_all_class)
    for i in range(len(classes)):
        result.update({classes[i]: i})
    return result
def decode_class(dict_class):
    result = {v:k for k, v in dict_class.items()}
    return result

def display_img(imgs_data, fig, axes):
    for i, img in enumerate(imgs_data):
        if type(img) == torch.Tensor:
            # img = torchvision.transforms.functional.rotate(img, 180)
            img = img.permute(1,2,0).numpy() # w h c
        axes[i].imshow(img)
        
    return fig, axes

def display_boundingbox(bboxes, classes, fig, axes):
    bboxes = ops.box_convert(bboxes, in_fmt='xyxy', out_fmt='xywh') #(xmin, ymin, xmax, ymax) to (x_center, y_center, width, height)
    class_index = 0
    for box in bboxes:
        x_center, y_center, width, height = box.numpy()

        #draw rectangle with edgecolor=red and no face color
        rect = patches.Rectangle((x_center, y_center), width, height,
                                 linewidth = 1, edgecolor = 'green',
                                 facecolor = 'none')
        
        axes.add_patch(rect)
        axes.text(x_center + 5, y_center + 20, classes[class_index],
                  bbox=dict(facecolor='yellow', alpha=0.5))
        class_index += 1
    return fig, axes

def generate_anchor_centers(output_size, scale):
    centers_x = (torch.arange(0, output_size[1]) + 0.5) * int(scale) # dùng + 0.5 để những anchor ở góc trái và góc trên không bị quá sát vào các biên
    centers_y = (torch.arange(0, output_size[0]) + 0.5) * int(scale)
    return centers_x, centers_y

def display_anchor_centers(centers_x, centers_y, fig, axes):
    for i in centers_x:
        for j in centers_y:
            axes.scatter(i, j, color='red', marker='+')
    return fig, axes

def display_anchor_boxes(centers, fig, axes):
    pass
