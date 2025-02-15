from skimage import io
from skimage.transform import resize
from utils import *
import os
import torch
from torchvision import ops
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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


class ProposalModule(nn.Module):
    def __init__(self):
        super(ProposalModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256,
                               kernel_size=3, padding=1) # 256x25x25 -> 256x25x25
        self.cls_logist = nn.Conv2d(in_channels=256, out_channels=9*2,
                                    kernel_size=1, stride=1, padding=0) # 256x25x25 -> 18x25x25
        self.bbox_reg = nn.Conv2d(in_channels=256, out_channels=9*4,
                                  kernel_size=1, stride=1, padding=0) # 256x25x25 -> 36x25x25
        
    def forward(self, feature_map, scale_feature2img = 8, positive_anchorbox_index=None,
                negative_anchorbox_index=None, positive_anchorbox_coordinates=None):
        x = self.conv1(feature_map)
        cls_logist = self.cls_logist(x)
        bbox_reg = self.bbox_reg(x)
        mode = 'train'
        if positive_anchorbox_index is None or\
            negative_anchorbox_index is None or\
                positive_anchorbox_coordinates is None:
            mode = 'eval'
        if mode == 'eval':
            return cls_logist.contiguous().view(-1, 2),\
                bbox_reg.contiguous().view(-1, 4)*scale_feature2img
        # get logist score of positive and negative anchor boxes
        cls_logist_pos = cls_logist.contiguous().view(-1, 2)[positive_anchorbox_index]
        cls_logist_neg = cls_logist.contiguous().view(-1, 2)[negative_anchorbox_index]

        # get offsets for positive anchor boxes
        offsets_pos = bbox_reg.contiguous().view(-1, 4)[positive_anchorbox_index]
        # generate proposal boxes
        proposals = generate_proposals(positive_anchorbox_coordinates,
                                       offsets_pos*scale_feature2img)

        return cls_logist_pos, cls_logist_neg,\
            offsets_pos*scale_feature2img, proposals

    
class RegionProposalNetwork(nn.Module):
    def __init__(self, feature_extractor, proposal_module):
        super(RegionProposalNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.proposal_module = proposal_module
        self.pos_threshold = 0.7
        self.neg_threshold = 0.2
        self.n_sample = 64
        self.pos_ratio = 0.5

        # weights for loss
        self.w_conf = 1
        self.w_reg = 5

    def forward(self, img_data, gt_boxes, gt_classes):
        '''
        input:
            img_data: tensor([1, chanel, width, height])
            gt_boxes: tensor([1, len_gt_boxes, 4])
            gt_classes: tensor([1, len_gt_boxes])
        output:
            total_rpn_loss: scalar number
            feature_map: tensor([1, chanel, width, height])
            proposals: tensor([N, 4])
            pos_index: tensor([N])
            gt_classes_img: tensor([N])
        '''
        feature_map = self.feature_extractor(img_data)

        # generate anchors
        _, _, feature_width, feature_height = feature_map.shape
        scale_center = img_data.shape[2]/feature_width
        centers_x, centers_y = generate_anchor_centers((feature_height, feature_width), scale_center)
        centers = [[x, y] for x in centers_x for y in centers_y]
        # generate anchor boxes
        all_bboxes = generate_all_anchor_boxes(centers, scale_center)
        all_bboxes_tensor = torch.stack(all_bboxes, dim=0)
        # calculate location, iou, label
        gt_appear_in_an_img = gt_boxes[0][torch.where(gt_boxes[0][:,0] != -1)[0]]
        all_location_boxes, all_iou_boxes, \
            all_label_boxes = calculate_location_iou_label(all_bboxes_tensor,
                                                           gt_appear_in_an_img.unsqueeze(0))
        '''
        all_location_boxes: tensor([tensor[x_top_left, y_top_left, dw, dh],...])
        dim = batch x N x 4, N is number of anchor boxes

        all_iou_boxes: tensor([tensor[iou1, iou2,...],...])
        dim = batch x N X M, M is number ground truth boxes in an image

        all_label_boxes: tensor([label1, label2,...])
        dim = batch x N

        batch currently is 1
        '''
        # with each gt box, find anchor box has max iou
        gt_argmax_iou = all_iou_boxes[0].argmax(dim=0)
        gt_max_iou = all_iou_boxes[0][gt_argmax_iou,
                                      torch.arange(gt_argmax_iou.shape[0])]
        
        # Find anchor boxes with corresponding iou score = corresponding gt_max_iou
        gt_argmax_iou = torch.where(all_iou_boxes[0] == gt_max_iou)[0]

        # for each anchor box in image, find gt box has max iou score with this anchor box
        anchor_boxes_argmax_iou = all_iou_boxes[0].argmax(dim=1)
        anchor_boxes_max_iou = all_iou_boxes[0][torch.arange(
            all_iou_boxes[0].shape[0]), anchor_boxes_argmax_iou]
        
        # for each anchor box, get coordinates of ground truth box has the max iou with this anchor box
        anchor_boxes_max_iou_coordinates = torch.stack([gt_appear_in_an_img[i]
                                                        for i in anchor_boxes_argmax_iou])
        # Get label for anchor box
        gt_classes_anc = gt_classes[0][torch.where(gt_classes[0][:] != -1)]
        gt_classes_img = torch.stack([gt_classes_anc[i]
                                      for i in anchor_boxes_argmax_iou])
        # assign label pos và neg for anchor box
        all_label_boxes[0][anchor_boxes_max_iou < self.neg_threshold] = 0
        all_label_boxes[0][gt_argmax_iou] = 1
        all_label_boxes[0][anchor_boxes_max_iou > self.pos_threshold] = 1

        # get n_sample anchor box
        n_pos = int(self.n_sample*self.pos_ratio)
        pos_index = torch.where(all_label_boxes[0] == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = torch.randperm(len(pos_index))[n_pos:]
            all_label_boxes[0][pos_index[disable_index]] = -1
        pos_index = torch.where(all_label_boxes[0] == 1)[0]

        n_neg = self.n_sample - len(torch.where(all_label_boxes[0] == 1)[0])
        neg_index = torch.where(all_label_boxes[0] == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = torch.randperm(len(neg_index))[n_neg:]
            all_label_boxes[0][neg_index[disable_index]] = -1
        neg_index = torch.where(all_label_boxes[0] == 0)[0]

        # calc offset for anchor box
        bboxes_pos_xyxy = ops.box_convert(all_location_boxes[0][all_label_boxes[0] == 1],
                                          in_fmt='xywh', out_fmt='xyxy')
        bboxes_neg_xyxy = ops.box_convert(all_location_boxes[0][all_label_boxes[0] == 0],
                                          in_fmt='xywh', out_fmt='xyxy')
        pos_score, neg_score, offset_pos, proposals = self.proposal_module(feature_map, scale_center,
                                                                           pos_index, neg_index,
                                                                           bboxes_pos_xyxy)
        
        gt_offset = calc_gt_offset(bboxes_pos_xyxy,
                                   anchor_boxes_max_iou_coordinates[pos_index])
        cls_loss = calc_cls_loss(pos_score, neg_score) # [1, 0] is negative, [0, 1] is positive
        reg_loss = calc_bbox_reg_loss(gt_offset, offset_pos)

        total_rpn_loss = self.w_conf* cls_loss + self.w_reg*reg_loss

        return total_rpn_loss, feature_map, proposals, \
            pos_index, gt_classes_img[pos_index]
    def inferences(self, img_data):
        feature_map = self.feature_extractor(img_data)
        # generate anchors
        _, _, feature_width, feature_height = feature_map.shape
        scale_center = img_data.shape[2]/feature_width
        centers_x, centers_y = generate_anchor_centers((feature_height,
                                                        feature_width), scale_center)
        centers = [[x, y] for x in centers_x for y in centers_y]
        # generate anchor boxes
        all_bboxes = generate_all_anchor_boxes(centers, scale_center)
        all_bboxes_tensor = torch.stack(all_bboxes, dim=0)
        # get conf_score and offsets
        conf_score, offsets = self.proposal_module(feature_map)
        # get proposals
        pos_index = torch.where(conf_score[:, 1] > conf_score[:, 0])[0]
        proposals = generate_proposals(all_bboxes_tensor.view(-1, 4)[pos_index],
                                       offsets[pos_index])
        return proposals, conf_score[pos_index], feature_map

class ClassificationModule(nn.Module):
    def __init__(self, n_classes,
                 roi_size=(7, 7), hidden_dim=256, feature_chanels=256):
        super(ClassificationModule, self).__init__()
        self.roi_size = roi_size
        self.avg_pool = nn.AvgPool2d(kernel_size=self.roi_size)
        self.fc = nn.Linear(feature_chanels, hidden_dim)
        self.cls_head = nn.Linear(hidden_dim, n_classes)
    def forward(self, feature_map, proposal_tensor_pos,
                gt_classes=None, img_size=(200,200)):
        '''
        gt_classes size: tensor([number of ground truth boxes])
        '''
        mode = 'train'
        if gt_classes is None:
            mode = 'eval'
        scale_img_feature = img_size[0]/feature_map.shape[2]
        proposal_feature_map = proposal_tensor_pos.detach()/scale_img_feature # size = N x 4 với N là số proposal
        proposal_feature_map = ops.clip_boxes_to_image(proposal_feature_map,
                                                       (feature_map.shape[3],
                                                        feature_map.shape[2]))
        '''
        Since region proposals have different size
        so need to change to the same size for calculate
        '''
        roi_out = ops.roi_pool(feature_map,
                               [proposal_feature_map.to(feature_map.device)],
                               self.roi_size)
        '''
        roi_out: tensor([N, C, roi_size, roi_size]) N is number of proposal,
        C is chanels of feature map
        '''
        roi_out = self.avg_pool(roi_out)

        #flatten output
        roi_out = roi_out.squeeze(-1).squeeze(-1)

        # pass roi_out through fully connected layer
        out = F.relu(self.fc(roi_out)) 

        # get the classification score
        cls_score = self.cls_head(out)

        if mode == 'eval':
            return cls_score
        
        # compute cross entropy loss
        cls_loss = F.cross_entropy(cls_score,
                                   gt_classes.long())
        return cls_loss

class TwoStageDetector(nn.Module):
    def __init__(self, feature_extractor,
                 proposal_module, n_classes):
        super(TwoStageDetector, self).__init__()
        self.rpn = RegionProposalNetwork(feature_extractor,
                                         proposal_module)
        self.classifier = ClassificationModule(n_classes)
    def forward(self, img_data, gt_boxes, gt_classes):
        rpn_loss, feature_map, proposals,\
            pos_index, gt_classes = self.rpn(img_data, gt_boxes, gt_classes)
        classifier_loss = self.classifier(feature_map, proposals, gt_classes)
        total_loss = rpn_loss + classifier_loss
        return total_loss
    def inferences(self, img_data):
        proposals, conf_score, feature_map = self.rpn.inferences(img_data)
        cls_score = self.classifier(feature_map, proposals)
        return proposals, conf_score, cls_score
        

class ObjectDetectionDataset(Dataset):
    def __init__(self, annotation_path, img_dir, img_size): # name2idx is encode classname as int, name2idx is encoder.
        self.annotation_path = annotation_path
        self.img_dir = img_dir
        self.img_size = img_size
        self.name2idx = None
        self.img_data_all,  self.gt_classes_all, \
            self.gt_bboxes_all = self.get_data()

    def __len__(self):
        return len(self.img_data_all)
    
    def __getitem__(self, index):
        return self.img_data_all[index], \
            self.gt_bboxes_all[index], self.gt_classes_all[index]
    
    def scale_bounding_box(self, scale_width, scale_height, boxes_img_raw):
        boxes_img = [[float(i[ind] * scale_width) if ind % 2 == 0 \
                      else float(i[ind] * scale_height)
                        for ind in range(len(i))] for i in boxes_img_raw]
        return boxes_img

    def get_data(self):
        img_data_all = []
        gt_idxs_all = []
        boxes_all_scale = []
        img_paths, gt_classes_all, \
            gt_boxes_all = parse_annotation(self.annotation_path,
                                            self.img_dir) # function from utils.py
        '''
        img_paths: [path1, path2, path3,...]
        gt_class_all: [[class1, class2,..], [class2,..], [class1, class3,..],...]
        gt_boxes_all: [[[x_min, y_min, x_max, y_max], [x_min, y_min, x_max, y_max],...],... ]
        '''
        
        for index, img_path in enumerate(img_paths):
            if (not img_path) or (not os.path.exists(img_path)):
                continue
            image = io.imread(img_path) # width, height, chanel
            old_width = image.shape[1]
            old_height = image.shape[0]
            # resize
            image = resize(image, self.img_size)

            # after resize image, we need to adjust bounding box data to fit the new image
            boxes_img_scale = self.scale_bounding_box(self.img_size[1]/old_width,
                                                      self.img_size[0]/old_height,
                                                      gt_boxes_all[index])
            boxes_all_scale.append(torch.Tensor(boxes_img_scale))

            # convert to tensor and reshape chanel, width, height
            image = torch.from_numpy(image).permute(2,0,1)
            img_data_all.append(image)

            #encode class name
            self.name2idx = encode_class(gt_classes_all) # make dictionary of class and index of it {'class': index}
            gt_idxs_all.append(torch.Tensor([self.name2idx[class_name]
                                             for class_name in \
                                                gt_classes_all[index]]))

        #padding bounding box and classes so that elements within them are the same size
        gt_idxs_all_padded = pad_sequence(gt_idxs_all, batch_first= True,
                                          padding_value=-1) # batch_first=True -> Batch, maxlenght, dim
        gt_boxes_all_padded = pad_sequence(boxes_all_scale, batch_first=True,
                                           padding_value=-1) # if batch_first=False -> maxlenght, batch, dim
        
        #stack all image to a Tensor
        imgs_stack = torch.stack(img_data_all, dim=0)
        # data in float32 type will save memory than float64(default)
        return imgs_stack.to(dtype=torch.float32), \
            gt_idxs_all_padded, gt_boxes_all_padded
