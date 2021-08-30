import torch
from torch.nn.modules.utils import _pair
from sagemakercv.core.structures.bounding_box import BoxList
from sagemakercv.detection.rpn.anchor_generator import AnchorGenerator, BufferList

class YOLOAnchorGenerator(AnchorGenerator):
    
    def __init__(self, strides, base_sizes):
        super(YOLOAnchorGenerator, self).__init__()
        self.strides = strides
        self.centers = [(stride / 2., stride / 2.)
                        for stride in self.strides]
        self.base_sizes = []
        num_anchor_per_level = len(base_sizes[0])
        for base_sizes_per_level in base_sizes:
            assert num_anchor_per_level == len(base_sizes_per_level)
            self.base_sizes.append(
                [_pair(base_size) for base_size in base_sizes_per_level])
        self.cell_anchors = BufferList(self.gen_base_anchors())

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.base_sizes)
    
    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return [base_anchors.size(0) for base_anchors in self.cell_anchors]

    def gen_base_anchors(self):
        """Generate base anchors.
        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_sizes_per_level in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(base_sizes_per_level,
                                                   center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self, base_sizes_per_level, center=None):
        """Generate base anchors of a single level.
        Args:
            base_sizes_per_level (list[tuple[int, int]]): Basic sizes of
                anchors.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.
        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        x_center, y_center = center
        base_anchors = []
        for base_size in base_sizes_per_level:
            w, h = base_size

            # use float anchor and the anchor's center is aligned with the
            # pixel center
            base_anchor = torch.Tensor([
                x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w,
                y_center + 0.5 * h
            ])
            base_anchors.append(base_anchor)
        base_anchors = torch.stack(base_anchors, dim=0)

        return base_anchors
    
    def forward(self, image_lists, feature_maps):
        anchors, batched_anchor_data = super(YOLOAnchorGenerator, self).forward(image_lists, feature_maps)
        anchors = [BoxList(torch.cat([i.bbox for i in j], dim=0), j[0].size) for j in anchors]
        return anchors, batched_anchor_data
        