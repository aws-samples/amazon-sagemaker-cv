import torch
from torch.nn import functional as F

from sagemakercv.layers import smooth_l1_loss
from sagemakercv.layers import GIoULoss
from sagemakercv.core.structures.bounding_box import BoxList
from sagemakercv.core.box_coder import BoxCoder
from sagemakercv.core.matcher import Matcher
from sagemakercv.core.structures.boxlist_ops import boxlist_iou, boxlist_iou_batched
from sagemakercv.core.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from sagemakercv.core.utils import cat
from torch.nn.utils.rnn import pad_sequence

class YoloLoss(object):
    """
    Computes the loss for Yolo.
    """

    def __init__(
        self, 
        anchor_generator,
        proposal_matcher, 
        fg_bg_sampler, 
        box_coder, 
        box_loss_type="SmoothL1Loss"
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.anchor_generator = anchor_generator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.giou_loss = GIoULoss(eps=1e-6, reduction="mean", loss_weight=10.0)
        self.box_loss_type = box_loss_type
        self.box_loss_func = torch.nn.SmoothL1Loss(reduction='none')
        self.class_loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
        self.object_loss_func = torch.nn.BCEWithLogitsLoss()
        self.mse_loss = torch.nn.MSELoss()
        
        self.lambda_xy, self.lambda_wh, self.lambda_obj, self.lambda_cls = 1.0/3, 1.0/3, 1.0/3, 2.5 #set weights
        
    def match_targets_to_proposals_batched(self, proposal, target):
        match_quality_matrix = boxlist_iou_batched(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix, batched=1)
        # Fast RCNN only need "labels" field for selecting the targets
        # how to do this for batched case?
        # target = target.copy_with_fields("labels")
        return matched_idxs
    
    def prepare_targets_batched(self, proposals, targets, target_labels):
        num_images = proposals.size(0)
        matched_idxs = self.match_targets_to_proposals_batched(proposals, targets)
        img_idx = torch.arange(num_images, device = proposals.device)[:, None]
        labels = target_labels[img_idx, matched_idxs.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)
        bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels.masked_fill_(bg_inds, 0)
        ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        labels.masked_fill_(ignore_inds, -1)

        matched_targets = targets[img_idx, matched_idxs.clamp(min=0)]
        
        if self.box_loss_type=="GIoULoss":
            regression_targets = matched_targets.view(-1,4)
        else:
            regression_targets = self.box_coder.encode(
                matched_targets.view(-1,4), proposals.view(-1,4)
            )
        return labels, regression_targets.view(num_images, -1, 4), matched_idxs
        
    def subsample(self, anchors, targets):
        target_boxes = pad_sequence([target.bbox for target in targets], batch_first = True, padding_value=-1)
        target_labels = pad_sequence([target.get_field("labels") for target in targets], batch_first = True, padding_value = -1)
        num_images = len(anchors)
        anchor_boxes = pad_sequence([anchor.bbox for anchor in anchors], batch_first=True, padding_value=-1)
        image_sizes = [anchor.size for anchor in anchors]
        labels, regression_targets, matched_idxs = self.prepare_targets_batched(anchor_boxes, target_boxes, target_labels)
        if num_images == 1:
            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels, is_rpn=0, objectness=None)
            # when num_images=1, sampled pos inds only has 1 item, so avoid copy in torch.cat
            pos_inds_per_image = [torch.nonzero(sampled_pos_inds[0]).squeeze(1)]
            neg_inds_per_image = [torch.nonzero(sampled_neg_inds[0]).squeeze(1)]
        else:
            sampled_pos_inds, sampled_neg_inds, num_pos_samples, num_neg_samples = self.fg_bg_sampler(labels, is_rpn=0, objectness=None)
            pos_inds_per_image = sampled_pos_inds.split(list(num_pos_samples))
            neg_inds_per_image = sampled_neg_inds.split(list(num_neg_samples))
        anchor_boxes = anchor_boxes.view(-1,4)
        regression_targets = regression_targets.view(-1,4)
        labels = labels.view(-1)
        matched_idxs = matched_idxs.view(-1)
        result_proposals = []
        for i in range(num_images):
            inds = torch.cat([pos_inds_per_image[i], neg_inds_per_image[i]])
            box = BoxList(anchor_boxes[inds], image_size = image_sizes[i])
            box.add_field("matched_idxs", matched_idxs[inds])
            box.add_field("regression_targets", regression_targets[inds])
            box.add_field("labels", labels[inds])
            box.add_field("inds", inds)
            result_proposals.append(box)
        self._proposals = result_proposals

        return result_proposals
    
    def get_targets(self, feature_maps, targets, proposals):
        targets_dict = {}
        features_cat = torch.cat([i.permute(0, 2, 3, 1).reshape(feature_maps[0].shape[0], -1, 85) \
                                  for i in feature_maps], dim=1).reshape(-1, 85)
        inds_cat = torch.cat([i.extra_fields['inds'] for i in proposals], dim=0)
        features_cat = features_cat[inds_cat]
        targets_dict['box_features_cat'] = features_cat[...,:4]
        targets_dict['obj_feature_cat'] = features_cat[..., 4]
        targets_dict['class_feature_cat'] = features_cat[..., 5:]
        targets_dict['target_boxes'] = torch.cat([i.extra_fields['regression_targets'] for i in proposals], dim=0)
        targets_dict['labels'] = torch.cat([i.extra_fields['labels'] for i in proposals], dim=0)
        targets_dict['mask'] = (torch.cat([i.extra_fields['matched_idxs'] for i in proposals], dim=0)>=0).to(torch.uint8)
        return targets_dict

    def box_loss(self, features, targets, mask):
        mask = mask.to(torch.float32) 
        features = features.to(torch.float32)
        targets = targets.to(torch.float32)
        
        fx = features[..., 0]          # Center x
        fy = features[..., 1]          # Center y
        fw = features[..., 2]                         # Width
        fh = features[..., 3]                         # Height
        
        tx = targets[..., 0]          # Center x
        ty = targets[..., 1]          # Center y
        tw = targets[..., 2]                        # Width
        th = targets[..., 3]                         # Height
        
        loss_x = self.mse_loss(fx*mask, tx*mask)
        loss_y = self.mse_loss(fy*mask, ty*mask)
        loss_w = self.mse_loss(fw*mask, tw*mask)
        loss_h = self.mse_loss(fh*mask, th*mask)
        
        return loss_x+loss_y+loss_w+loss_h
        
    
    def object_loss(self, objectness, mask):
        obj_loss = self.object_loss_func(objectness, mask.to(torch.float)) 
#         obj_loss = self.object_loss_func(objectness[mask==1], mask[mask==1]) 
        return obj_loss
    
    def class_loss(self, class_features, labels, mask):
        minus = torch.ones_like(labels)
        reduced_label = torch.sub(labels, minus, alpha=1)
        labels = reduced_label.clamp(min=0)
        return self.class_loss_func(class_features[mask == 1], labels[mask == 1])
    
    def __call__(self, images, feature_maps, target_bboxes):
        anchors, anchor_meta_data = self.anchor_generator(images, feature_maps)
        proposals = self.subsample(anchors, target_bboxes)
        parsed_targets = self.get_targets(feature_maps, target_bboxes, proposals)
        losses = {}
        losses['box_loss'] = self.box_loss(parsed_targets['box_features_cat'],
                                           parsed_targets['target_boxes'],
                                           parsed_targets['mask']) * 1.
        losses['class_loss'] = self.class_loss(parsed_targets['class_feature_cat'],
                                               parsed_targets['labels'],
                                               parsed_targets['mask']) * 2.
        losses['object_loss'] = self.object_loss(parsed_targets['obj_feature_cat'],
                                                 parsed_targets['mask']) * 5.
        losses['total_loss'] = sum(losses.values())
        return losses, parsed_targets
