import xml.etree.ElementTree as ET
import os
import matplotlib.patches as patches
import torch
from torchvision import ops
import torch.nn.functional as F
import math

def parse_annotation(annotation_path, imgs_dir):
    '''
    input:
        annotation_path: path to annotation folder
        imgs_dir: path to image folder
    output:
        imgs_list_dir: list of image directory, len = N (N is number of images)
        gt_class_all: list of list of class of bounding box, size = (N, M) with M is number of bounding box in each image,
                        M different in each image, N as above
        gt_boxes_all: list of list of bounding box, size = (N, M, 4), M as above, 4 is (xmin, ymin, xmax, ymax)

    '''
    #traverse xml
    gt_class_all = []
    gt_boxes_all = []
    imgs_list_dir = []
    count = 0
    for xml_file in os.listdir(annotation_path):
        if count >= 1000:
            break
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
        count += 1
    return imgs_list_dir, gt_class_all, gt_boxes_all

def encode_class(gt_classes_all):
    '''
    input:
        gt_classes_all: list of all classes in all images,
        size: (N, M) with N is number of images, M is number of bounding box in each image
    output:
        dict: {'class_name1': index1, 'class_name2': index2, ...}
    '''
    result = dict()
    list_all_class = [item for list_item in gt_classes_all for item in list_item]
    set_all_class = set(list_all_class) # remove duplicate
    classes = list(set_all_class)
    for i in range(len(classes)):
        result.update({classes[i]: i})
    return result
def decode_class(dict_class):
    '''
    input:
        dict: {'class_name1': index1, 'class_name2': index2, ...}
    output:
        dict: {index1: 'class_name1', index2: 'class_name2', ...}
    '''
    result = {v:k for k, v in dict_class.items()}
    return result

def display_img(imgs_data, fig, axes):
    '''
    input:
        imgs_data: list of image data, size = N (N images)
        fig, axes: figure and axes of matplotlib
    output:
        fig, axes: figure and axes of matplotlib
    '''
    for i, img in enumerate(imgs_data):
        if type(img) == torch.Tensor:
            img = img.permute(1,2,0).numpy() # w h c
        axes[i].imshow(img)
    
    return fig, axes

def display_boundingbox(bboxes, classes, fig, axes):
    '''
    display bounding box on an image
    input:
        bboxes: list of bounding box of an image, size = (M, 4)
                with M is number of bounding box in an image
        classes: list class of bounding box, size = M
        fig, axes: figure and axes of matplotlib
    output:
        fig, axes: figure and axes of matplotlib
    '''
    #(xmin, ymin, xmax, ymax) to (x_top_left, y_top_left, width, height)
    bboxes = ops.box_convert(bboxes, in_fmt='xyxy',
                             out_fmt='xywh') 
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
    '''
    input:
        output_size: (height, width) of feature map
        scale: image_size / feature_map size
    output:
        centers_x, centers_y: list of x, y of anchor centers in image
    '''
    centers_x = (torch.arange(0, output_size[1]) + 0.5) * int(scale)
    # dùng + 0.5 để những anchor ở góc trái và góc trên không bị quá sát vào các biên
    centers_y = (torch.arange(0, output_size[0]) + 0.5) * int(scale)
    return centers_x, centers_y

def display_anchor_centers(centers_x, centers_y, fig, axes):
    '''
    input: 
        centers_x, centers_y:
            center_x_size: (n, ) with n is number of center width
            center_y_size: (m, ) with m is number of center height
        fig, axes: figure and axes of matplotlib
    output:
        fig, axes: figure and axes of matplotlib
    '''
    for i in centers_x:
        for j in centers_y:
            axes.scatter(i, j,
                         color='red',
                         marker='+')
    return fig, axes

def generate_all_anchor_boxes(centers, scales_img):
    '''
    with each anchor center, create 9 anchor boxes: size is 4 x 4, 8 x 8, 16 x 16 pixel of feature map 
            and have 3 scales ratio 1:1, 1:2, 2:1 corresponding height:width with each size of bounding box
    input:
        centers: (centers_x, centers_y)
        scales_img: scale of image to feature map (img_size / feature_map_size)
    output:
        anchor_boxes: list of tensor with size (9, 4) with 9 is number of anchor boxes of each anchor center
            len of list is number of anchor center
    '''
    ratios = [0.5, 1, 2] # 1:2, 1:1, 2:1 width:height
    scales = [4, 8, 16]
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
                list_boxes.append(torch.Tensor([x_top_left, y_top_left,
                                                width, height]))
        if len(list_boxes) > 0:
            anchor_boxes.append(torch.stack(list_boxes))
    return anchor_boxes

def calculate_IOU( anchor_box, ground_truth_box):
    '''
    Calculate IOU between an anchor box and a ground truth box
    input:
        anchor_box: (x_top_left, y_top_left, width, height)
        ground_truth_box: (x_min, y_min, x_max, y_max)
    output:
        iou score
    '''
    anchor_box_xyxy = ops.box_convert(anchor_box, in_fmt='xywh', out_fmt='xyxy')
    x_min_anchor, y_min_anchor,\
        x_max_anchor, y_max_anchor = anchor_box_xyxy.squeeze(0).numpy()
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

def calculate_location_iou_label(anchor_boxes, ground_truth_boxes_batch):

    '''
    input:
        anchor_boxes: Nx9x4; N is number of anchor, M is number of anchor boxes of each anchor
        gt_boxes_batch: batchxMx4; M is number of ground truth box of each image
    output:
        1. localtion: [[[dx, dy, dw, dh], ...]], size: (batch, N, 4), N is number of anchor boxes of all anchor
        2. iou: [[[iou_gt1, iou_gt2,...], ...]], size: (batch, N, M), N as above, M is number of ground truth box of an image
        3. label: [[label1, ...]], size: (batch, N,), N as above
        batch is currently 1
    '''
    location_box_batch = torch.zeros(ground_truth_boxes_batch.size(0),
                                     anchor_boxes.size(0)*anchor_boxes.size(1), 4)
    iou_boxes_batch = torch.zeros(ground_truth_boxes_batch.size(0),
                                  anchor_boxes.size(0)*anchor_boxes.size(1),
                                  ground_truth_boxes_batch.size(1))
    label_boxes_batch = torch.zeros(ground_truth_boxes_batch.size(0),
                                    anchor_boxes.size(0)*anchor_boxes.size(1))
    cnt = 0
    for ground_truth_boxes in ground_truth_boxes_batch:
        location_box = torch.zeros(anchor_boxes.size(0)*anchor_boxes.size(1), 4)
        iou_boxes = torch.zeros(anchor_boxes.size(0)*anchor_boxes.size(1),
                                ground_truth_boxes_batch.size(1))
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


def generate_proposals(anchors, offsets):

    '''
    generate proposals region by adjusting positive anchor boxes by offsets, offsets is what model need to learn
    input: N is number of anchor boxes
        anchors: ((x, y, x, y),..) Nx4
        offsets: ((tx, ty, tw, th),..) Nx4
    output:
        proposals: ((x, y, x, y),..) Nx4
    '''
    proposals = torch.zeros(anchors.size()).to(anchors.device)
    # check if anchors is empty
    if anchors.size(0) == 0:
        return proposals
    # convert anchor boxes from xyxy to xywh
    anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='xywh')
    offsets = offsets.to(anchors.device)
    # generate proposals
    proposals[:, 0] = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
    proposals[:, 1] = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
    proposals[:, 2] = anchors[:, 2] * torch.exp(offsets[:, 2])
    proposals[:, 3] = anchors[:, 3] * torch.exp(offsets[:, 3])

    # convert proposals from xywh to xyxy
    proposals = ops.box_convert(proposals, in_fmt='xywh', out_fmt='xyxy')

    return proposals


def calc_gt_offset(pos_anc_coords, gt_bbox_mapping):
    '''
    calculate offset between positive anchor boxes and the gt boxes it represents
    input:
        pos_anc_coords: tensor (n_pos, 4)
        gt_bbox_mapping: tensor (n_pos, 4)
    output:
        gt_offsets: tensor (n_pos, 4)
    '''
    pos_anc_coords = pos_anc_coords.to(gt_bbox_mapping.device)
    gt_offsets = torch.zeros_like(pos_anc_coords).to(gt_bbox_mapping.device)
    width_anchor_boxes = pos_anc_coords[:, 2]
    height_anchor_boxes = pos_anc_coords[:, 3]
    centers_x = pos_anc_coords[:, 0] + width_anchor_boxes / 2
    centers_y = pos_anc_coords[:, 1] + height_anchor_boxes / 2

    width_gt_boxes = gt_bbox_mapping[:, 2]
    height_gt_boxes = gt_bbox_mapping[:, 3]
    centers_x_gt = gt_bbox_mapping[:, 0] + width_gt_boxes / 2
    centers_y_gt = gt_bbox_mapping[:, 1] + height_gt_boxes / 2

    # adjust width and height of anchor boxes avoid log(0)
    eps = torch.finfo(width_anchor_boxes.dtype).eps
    width_anchor_boxes = torch.clamp_min(width_anchor_boxes, eps) # clamp_min: keep min value is là eps, avoid = 0
    height_anchor_boxes = torch.clamp_min(height_anchor_boxes, eps)

    gt_offsets[:,0] = (centers_x_gt - centers_x) / width_anchor_boxes
    gt_offsets[:,1] = (centers_y_gt - centers_y) / height_anchor_boxes
    gt_offsets[:,2] = torch.log(width_gt_boxes / width_anchor_boxes)
    gt_offsets[:,3] = torch.log(height_gt_boxes / height_anchor_boxes)

    return gt_offsets
    

def calc_cls_loss(conf_scores_pos, conf_score_neg):
    '''
    calculate classification loss
    input:
        conf_scores_pos: tensor (n_pos, 2)
        conf_score_neg: tensor (n_neg, 2)
    output:
        loss: tensor (1)
    '''
    target_pos = torch.ones_like(conf_scores_pos)
    target_neg = torch.zeros_like(conf_score_neg)

    target = torch.cat((target_pos, target_neg))
    input = torch.cat((conf_scores_pos, conf_score_neg))

    loss = F.binary_cross_entropy_with_logits(input, target,
                                              reduction='sum')
    return loss

def calc_bbox_reg_loss(gt_offset, reg_offset_pos):
    '''
    calculate bounding box regression loss
    input:
        gt_offset: tensor (n_pos, 4)
        reg_offset_pos: tensor (n_pos, 4)
    output:
        loss: tensor (1)
    '''
    loss = F.smooth_l1_loss(reg_offset_pos, gt_offset,
                            reduction='sum')
    return loss
