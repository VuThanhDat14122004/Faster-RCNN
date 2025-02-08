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
import math

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
            img = img.permute(1,2,0).numpy() # w h c
        axes[i].imshow(img)
    
    return fig, axes

def display_boundingbox(bboxes, classes, fig, axes):
    bboxes = ops.box_convert(bboxes, in_fmt='xyxy', out_fmt='xywh') #(xmin, ymin, xmax, ymax) to (x_top_left, y_top_left, width, height)
    class_index = 0
    # bboxes là tensor có kích thước (n, 4) với n là số bounding box có trong ảnh
    for box in bboxes:
        x_top_left, y_top_left, width, height = box.numpy()
        rect = patches.Rectangle((int(x_top_left), int(y_top_left)), int(width), int(height),
                                 linewidth = 1, edgecolor = 'green',
                                 facecolor = 'none')
        
        axes.add_patch(rect)
        axes.text(int(x_top_left), int(y_top_left), classes[class_index],
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

# def generate_anchor_boxes(centers, imgsize, scales_img): # centers = (centers_x, centers_y)
#     '''
#     với mỗi anchor center, ta sẽ tạo ra 9 anchor box : kích thước là 2 x 2, 4 x 4, 6 x 6 pixel của feature map và với mỗi kích thước
#     thì có 3 tỉ lệ là 1:1, 1:2, 2:1 tương ứng với height và width
#     '''
#     ratios = [0.5, 1, 2] # 1:2, 1:1, 2:1 width:height
#     scales = [2, 4, 6]
#     scales = [scale * scales_img for scale in scales]

#     anchor_boxes = []
#     for center in centers:
#         list_boxes = []
#         for scale in scales:
#             for ratio in ratios:
#                 height = int(math.sqrt(math.pow(scale, 2) / ratio))
#                 width = int(math.pow(scale, 2) / height)
#                 x_top_left = int(center[0] - width / 2)
#                 y_top_left = int(center[1] - height / 2)
#                 if check_valid_anchor([x_top_left, y_top_left, width, height], imgsize):
#                     list_boxes.append(torch.Tensor([x_top_left, y_top_left, width, height]))
#         if len(list_boxes) > 0:
#             anchor_boxes.append(torch.stack(list_boxes))
#     return anchor_boxes # list chứa các tensor có kích thước (n, 4) với n là số anchor box hợp lệ của mỗi anchor

def generate_all_anchor_boxes(centers, scales_img): # centers = (centers_x, centers_y)
    '''
    với mỗi anchor center, ta sẽ tạo ra 9 anchor box : kích thước là 2 x 2, 4 x 4, 6 x 6 pixel của feature map và với mỗi kích thước
    thì có 3 tỉ lệ là 1:1, 1:2, 2:1 tương ứng với height và width
    '''
    ratios = [0.5, 1, 2] # 1:2, 1:1, 2:1 width:height
    scales = [2, 4, 6]
    scales = [scale * scales_img for scale in scales]

    anchor_boxes = []
    for center in centers:
        list_boxes = []
        for scale in scales:
            for ratio in ratios:
                height = int(math.sqrt(math.pow(scale, 2) / ratio))
                width = int(math.pow(scale, 2) / height)
                x_top_left = int(center[0] - width / 2)
                y_top_left = int(center[1] - height / 2)
                list_boxes.append(torch.Tensor([x_top_left, y_top_left, width, height]))
        if len(list_boxes) > 0:
            anchor_boxes.append(torch.stack(list_boxes))
    return anchor_boxes # list chứa các tensor có kích thước (n, 4) với n là tất cả số anchor box của mỗi anchor

def check_valid_anchor(anchor_box, imgsize):
    x_top_left, y_top_left, width, height = anchor_box
    if x_top_left >= 1 and y_top_left >=1 and x_top_left + width < imgsize[1]  and y_top_left + height < imgsize[0]:
        return True
    return False

def calculate_IOU( anchor_box, ground_truth_box):
    '''
    anchor_box: (x_top_left, y_top_left, width, height)
    ground_truth_box: (x_min, y_min, x_max, y_max)
    '''
    anchor_box_xyxy = ops.box_convert(anchor_box, in_fmt='xywh', out_fmt='xyxy') # cần nhân lên với scale để lấy tọa độ thực tế
    x_min_anchor, y_min_anchor, x_max_anchor, y_max_anchor = anchor_box_xyxy.squeeze(0).numpy()
    x_min_gt, y_min_gt, x_max_gt, y_max_gt = ground_truth_box
    x_min_inter = max(x_min_anchor, x_min_gt)
    y_min_inter = max(y_min_anchor, y_min_gt)
    x_max_inter = min(x_max_anchor, x_max_gt)
    y_max_inter = min(y_max_anchor, y_max_gt)
    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
    anchor_area = (x_max_anchor - x_min_anchor) * (y_max_anchor - y_min_anchor)
    gt_area = (x_max_gt - x_min_gt) * (y_max_gt - y_min_gt)
    union_area = anchor_area + gt_area - inter_area
    iou = inter_area / union_area
    return iou
# def get_bboxes_negative_and_positive(anchor_boxes, ground_truth_boxes, threshold = 0.5):

#     '''
#     lấy ra các anchor box có IOU lớn hơn threshold với ground truth box
#     '''
#     bboxes_postive = []
#     bboxes_negative = []
#     iou_pos_list = []
#     for ground_truth_box in ground_truth_boxes:
#         ious_1_object = []
#         bboxes_pos_1_object = []
#         bboxes_neg_1_object = []
#         for anchor_box in anchor_boxes:
#             for box in anchor_box:
#                 iou = calculate_IOU(box, ground_truth_box)
#                 if iou > threshold:
#                     bboxes_pos_1_object.append(box)
#                     ious_1_object.append(iou)
#                 else:
#                     bboxes_neg_1_object.append(box)
#         iou_pos_list.append(ious_1_object)
#         try:
#             bboxes_postive.append(torch.stack(bboxes_pos_1_object))
#         except:
#             bboxes_postive.append(torch.Tensor())
#         try:
#             bboxes_negative.append(torch.stack(bboxes_neg_1_object))
#         except:
#             bboxes_negative.append(torch.Tensor())
#     return bboxes_postive, bboxes_negative, iou_pos_list

def calculate_location_iou_label(anchor_boxes, ground_truth_boxes_batch):

    '''
    input:
        anchor_boxes: NxMx4; N là số anchor, M là số anchor box của mỗi anchor
        gt_boxes_batch: batchxMx4; M là số ground truth box của mỗi ảnh
    return 3 list
    1. localtion: [[[dx, dy, dw, dh], ...]], size: (batch, N, 4), với N là số anchor box
    2. iou: [[[iou_gt1, iou_gt2,...], ...]], size: (batch, N, M), với N như trên, M là số ground truth box trong ảnh
    3. label: [[label1, ...]], size: (batch, N,), với N như trên
    '''
    location_box_batch = torch.zeros(ground_truth_boxes_batch.size(0), anchor_boxes.size(0)*anchor_boxes.size(1), 4)
    iou_boxes_batch = torch.zeros(ground_truth_boxes_batch.size(0), anchor_boxes.size(0)*anchor_boxes.size(1), ground_truth_boxes_batch.size(1))
    label_boxes_batch = torch.zeros(ground_truth_boxes_batch.size(0), anchor_boxes.size(0)*anchor_boxes.size(1))
    cnt = 0
    for ground_truth_boxes in ground_truth_boxes_batch:
        location_box = torch.zeros(anchor_boxes.size(0)*anchor_boxes.size(1), 4)
        iou_boxes = torch.zeros(anchor_boxes.size(0)*anchor_boxes.size(1), ground_truth_boxes_batch.size(1))
        sub_cnt = 0
        for list_anchor_box in anchor_boxes:
            for anchor_box in list_anchor_box:
                location_box[sub_cnt, :] = anchor_box
                iou_box = []
                for ground_truth_box in ground_truth_boxes:
                    if ground_truth_box[0] == -1:
                        iou_box.append(0)
                        continue
                    iou_box.append(calculate_IOU(anchor_box, ground_truth_box))
                iou_boxes[sub_cnt, :] = torch.tensor(iou_box)
                sub_cnt += 1
        location_box_batch[cnt, :] = location_box
        iou_boxes_batch[cnt, :] = iou_boxes
        label_boxes_batch[cnt, :] = torch.tensor([-1 for id in iou_boxes])
        cnt += 1
    return location_box_batch, iou_boxes_batch, label_boxes_batch


# def gen_anchor_boxes_base(centers_x, centers_y, scales_img, imgsize):

#     ratios = [0.5, 1, 2] # 1:2, 1:1, 2:1 width:height
#     scales = [2, 4, 6]
#     scales = [scale * scales_img for scale in scales]
#     n_anchor_boxes_each_center = len(ratios) * len(scales)
#     anchor_boxes_base = torch.zeros(1, centers_x.size(0), centers_y.size(0), n_anchor_boxes_each_center, 4)
#     for ix, xc in enumerate(centers_x):
#         for iy, yc in enumerate(centers_y):
#             anchor_boxes = torch.zeros(n_anchor_boxes_each_center, 4)
#             cnt = 0
#             for scale in scales:
#                 for ratio in ratios:
#                     height = int(math.sqrt(math.pow(scale, 2) / ratio))
#                     width = int(math.pow(scale, 2) / height)
#                     x_top_left = int(xc - width / 2)
#                     y_top_left = int(yc - height / 2)
#                     x_bot_right = int(xc + width / 2)
#                     y_bot_right = int(yc + height / 2)
#                     anchor_boxes[cnt, :] = torch.Tensor([x_top_left, y_top_left,
#                                                          x_bot_right, y_bot_right])
#                     cnt += 1
#             anchor_boxes_base[:, ix, iy, :, :] = ops.clip_boxes_to_image(anchor_boxes, imgsize)
#     return anchor_boxes_base # size: (1, centers_x.size(0), centers_y.size(0), n_anchor_boxes_each_center, 4)

def generate_proposals(anchors, offsets):

    '''
    sinh ra proposals region bằng cách điều chỉnh anchor boxes (positive) bằng offsets (cái mà mô hình cần học)
    input:
        anchors: ((x, y, x, y),..) Nx4
        offsets: ((tx, ty, tw, th),..) Nx4
    output:
        proposals: ((x, y, x, y),..) Nx4
    '''

    # convert anchor boxes from xyxy to xywh
    anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='xywh')
    
    # generate proposals
    proposals = torch.zeros(anchors.size())
    proposals[:, 0] = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
    proposals[:, 1] = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
    proposals[:, 2] = anchors[:, 2] * torch.exp(offsets[:, 2])
    proposals[:, 3] = anchors[:, 3] * torch.exp(offsets[:, 3])

    # convert proposals from xywh to xyxy
    proposals = ops.box_convert(proposals, in_fmt='xywh', out_fmt='xyxy')

    return proposals


def calc_gt_offset(pos_anc_coords, gt_bbox_mapping): # n_pos x 4
    '''
    tính offset giữa positive anchor boxes với gt boxes mà nó đại diện
    input:
        pos_anc_coords: tensor (n_pos, 4)
        gt_bbox_mapping: tensor (n_pos, 4)
    output:
        gt_offsets: tensor (n_pos, 4)
    '''
    gt_offsets = torch.zeros_like(pos_anc_coords)
    width_anchor_boxes = pos_anc_coords[:, 2]
    height_anchor_boxes = pos_anc_coords[:, 3]
    centers_x = pos_anc_coords[:, 0] + width_anchor_boxes / 2
    centers_y = pos_anc_coords[:, 1] + height_anchor_boxes / 2

    width_gt_boxes = gt_bbox_mapping[:, 2]
    height_gt_boxes = gt_bbox_mapping[:, 3]
    centers_x_gt = gt_bbox_mapping[:, 0] + width_gt_boxes / 2
    centers_y_gt = gt_bbox_mapping[:, 1] + height_gt_boxes / 2

    # Chỉnh lại width và height của anchor boxes để tránh log(0)
    eps = torch.finfo(width_anchor_boxes.dtype).eps
    width_anchor_boxes = torch.clamp_min(width_anchor_boxes, eps) # clamp_min: giữ giá trị nhỏ nhất là eps, tránh = 0
    height_anchor_boxes = torch.clamp_min(height_anchor_boxes, eps)

    gt_offsets[:,0] = (centers_x_gt - centers_x) / width_anchor_boxes
    gt_offsets[:,1] = (centers_y_gt - centers_y) / height_anchor_boxes
    gt_offsets[:,2] = torch.log(width_gt_boxes / width_anchor_boxes)
    gt_offsets[:,3] = torch.log(height_gt_boxes / height_anchor_boxes)

    return gt_offsets
    

def calc_cls_loss(conf_scores_pos, conf_score_neg):
    target_pos = torch.ones_like(conf_scores_pos)
    target_neg = torch.zeros_like(conf_score_neg)

    target = torch.cat((target_pos, target_neg))
    input = torch.cat((conf_scores_pos, conf_score_neg))

    loss = F.binary_cross_entropy_with_logits(input, target, reduction='sum')
    return loss

def calc_bbox_reg_loss(gt_offset, reg_offset_pos):
    loss = F.smooth_l1_loss(reg_offset_pos, gt_offset, reduction='sum')
    return loss