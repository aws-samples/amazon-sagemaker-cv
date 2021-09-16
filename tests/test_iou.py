import torch
from collections import OrderedDict
from smcv_utils import _C
from torch.nn.utils.rnn import pad_sequence

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

anchors = []
inds_inside = []
for cell_anchor, feature_size, image_size, stride \
    in zip(cell_anchors, feature_sizes, image_sizes, strides):
    anchor, inds = _C.anchor_generator(*image_size, 
                                        feature_size, 
                                        cell_anchors[cell_anchor], 
                                        stride, straddle_thresh)
    anchors.append(anchor)
    inds_inside.append(inds)
    
anchor_boxes = torch.cat(anchors, dim=0).unsqueeze(dim=0)
anchor_visibility = torch.cat(inds_inside, dim=0).unsqueeze(dim=0)


targets = [torch.tensor([[186.0830, 146.0500, 295.3147, 254.1167],
                         [353.8781, 158.8833, 395.7352, 275.8167]]).to('cuda'),
           torch.tensor([[ 691.8182,  376.1677,  810.4888,  446.7545],
                         [ 806.5865,  614.0599,  819.4183,  636.6467],
                         [ 670.5115,  418.1078,  879.6753,  514.3713],
                         [ 691.5308,  374.4670,  810.3211,  448.0000],
                         [ 273.1315,  537.1018,  538.4346,  797.6048],
                         [ 974.0468,  575.5928, 1193.8878,  785.1257],
                         [ 233.0320,  666.4670,  335.0164,  797.6048],
                         [ 489.3336,  107.2814, 1015.8460,  784.0718]]).to('cuda'),
           torch.tensor([[510.5305,  70.1077, 974.2625, 789.1335],
                         [451.4422,  21.2272, 740.3076, 617.0867],
                         [243.5469, 697.3303, 287.1230, 742.9134],
                         [ 67.0878, 688.5059, 110.1956, 735.6628],
                         [185.1518, 368.5433, 404.4002, 683.0727],
                         [604.5021, 191.9438, 645.3806, 296.9180],
                         [614.7311, 606.7634, 646.9916, 641.4801],
                         [629.5874, 661.8267, 667.9367, 702.3701],
                         [353.0493, 748.1406, 401.0280, 796.9087],
                         [531.6815, 196.8150, 566.6586, 228.2717],
                         [503.4114, 282.3419, 544.2336, 329.2178],
                         [479.5438, 594.3420, 508.3573, 626.2670],
                         [  0.0000, 373.2646, 706.9979, 787.2413]]).to('cuda'),
           torch.tensor([[163.2066, 501.4614, 573.6593, 746.0054]]).to('cuda')]

targets = pad_sequence(targets, batch_first=True, padding_value=-1)

def test_iou():
    iou = _C.box_iou(torch.cat([anchor_boxes]*4, dim=0), targets)
    assert tuple(iou.shape)==(4, 242991, 13)
    
if __name__=='__main__':
    test_iou()
