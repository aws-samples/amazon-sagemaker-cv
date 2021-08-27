# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor, make_cascade_box_predictor
from .inference import make_roi_box_post_processor
from sagemakercv.detection import registry
from .loss import make_roi_box_loss_evaluator, make_roi_cascade_loss_evaluator

@registry.ROI_BOX_HEAD.register("StandardBoxHead")
class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        results = self.loss_evaluator(
            [class_logits.float()], [box_regression.float()]
        )

        if len(results) > 2:
            loss_dict = dict(loss_classifier=results[0], loss_box_reg=results[1], loss_carl=results[2])
        else:
            loss_dict = dict(loss_classifier=results[0], loss_box_reg=results[1])
        return (
            x,
            proposals,
            loss_dict,
        )
    
@registry.ROI_BOX_HEAD.register("CascadeBoxHead")
class CascadeBoxHead(torch.nn.Module):
    """
    Cascade Box Head class.
    """
    def __init__(self, cfg):
        super(CascadeBoxHead, self).__init__()
        self.stages = cfg.MODEL.ROI_HEADS.CASCADE_STAGES
        self.stage_weights = cfg.MODEL.ROI_HEADS.CASCADE_STAGE_WEIGHTS
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictors = make_cascade_box_predictor(cfg)
        self.loss_evaluators = make_roi_cascade_loss_evaluator(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.
        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            loss_dict = {}
            for stage in range(self.stages):
                with torch.no_grad():
                    proposals = self.loss_evaluators[stage].subsample(proposals, targets)
                x = self.feature_extractor(features, proposals)
                class_logits, box_regression = self.predictors[stage](x)
                results = self.loss_evaluators[stage](
                    [class_logits.float()], [box_regression.float()]
                )
                loss_dict.update({f"loss_classifier_{stage}": results[0] * self.stage_weights[stage],
                                  f"loss_box_reg_{stage}": results[1] * self.stage_weights[stage]})
                if len(results) > 2:
                    loss_dict.update({f"loss_carl_{stage}": results[2]})
                if stage < self.stages - 1:
                    proposals = self.loss_evaluators[stage].refine(proposals, box_regression)
            return (
                x,
                proposals,
                loss_dict,
                )
        else:
            for stage in range(self.stages):
                x = self.feature_extractor(features, proposals)
                class_logits, box_regression = self.predictors[stage](x)
                if stage < self.stages - 1:
                    proposals = self.loss_evaluators[stage].refine(proposals, box_regression)
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    func = registry.ROI_BOX_HEAD[cfg.MODEL.ROI_BOX_HEAD.TYPE]
    return func(cfg)
