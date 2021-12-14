import torch
from collections import OrderedDict
from smcv_utils import _C

# Anchor generator tests
cell_anchors = OrderedDict()
cell_anchors[0] = torch.tensor([[-22., -10., 25., 13.], [-14., -14.,  17.,  17.], [-10., -22.,  13.,  25.]]).to('cuda')
cell_anchors[1] = torch.tensor([[-40., -20.,  47.,  27.], [-28., -28.,  35.,  35.], [-20., -44.,  27.,  51.]]).to('cuda')
cell_anchors[2] = torch.tensor([[-84., -40.,  99.,  55.], [-56., -56.,  71.,  71.], [-36., -80.,  51.,  95.]]).to('cuda')
cell_anchors[3] = torch.tensor([[-164.,  -72.,  195.,  103.], [-112., -112.,  143.,  143.], [ -76., -168.,  107.,  199.]]).to('cuda')
cell_anchors[4] = torch.tensor([[-332., -152.,  395.,  215.], [-224., -224.,  287.,  287.], [-148., -328.,  211.,  391.]]).to('cuda')

straddle_thresh = 0

feature_sizes = [[200, 304], [100, 152], [50, 76], [25, 38], [13, 19]]

image_sizes =  torch.tensor([[1199,  800], 
                             [1132,  800], 
                             [1196,  800], 
                             [1066,  800], 
                             [1066,  800], 
                             [1003,  800], 
                             [1066,  800], 
                             [1199,  800]], dtype=torch.int32).to('cuda')

strides = (4, 8, 16, 32, 64)

anchor_sums = [367353600, 91838400., 22959600., 5739900., 1516086.]
inds_inside_sums = [112304, 26398, 5698, 1004, 90]

def test_anchor_generator():
    for cell_anchor, feature_size, image_size, stride, anchor_sum, ind_sum \
        in zip(cell_anchors, feature_sizes, image_sizes, strides, anchor_sums, inds_inside_sums):
        anchors, inds_inside = _C.anchor_generator(*image_size, 
                                            feature_size, 
                                            cell_anchors[cell_anchor], 
                                            stride, straddle_thresh)
        assert anchors.sum()==anchor_sum
        assert inds_inside.sum()==ind_sum

if __name__=='__main__':
    test_anchor_generator()
