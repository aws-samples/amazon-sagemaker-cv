import torch
from collections import OrderedDict
from smcv_utils import _C

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

image_sizes_wh = [tuple(i) for i in image_sizes]

strides = (4, 8, 16, 32, 64)

fmap_size_cat = torch.tensor([[304, 200],
                                [152, 100],
                                [ 76,  50],
                                [ 38,  25],
                                [ 19,  13]]).to('cuda')

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

objectness = [torch.rand([1, 3, *i[-2:]]).to('cuda') for i in fmap_size_cat]
rpn_box_regression = [torch.rand([1, 12, *i[-2:]]).to('cuda') for i in fmap_size_cat]

N = 1
A = 3
H_max = 304
W_max = 200
num_fmaps = 5
num_anchors_per_level = torch.tensor([182400,  45600,  11400,   2850,    741]).to('cuda')
num_max_proposals = [2000 for i in range(N*num_fmaps)]
num_max_props_tensor = [2000 for i in range(N*num_fmaps)]


batched_objectness_tensor = -1e6 * torch.ones([num_fmaps, N, A * H_max * W_max],  \
                                                        dtype = objectness[0].dtype, device=objectness[0].device)
batched_regression_tensor = -1 * torch.ones([num_fmaps, N, 4 * A * H_max * W_max], \
                                                        dtype = objectness[0].dtype, device=objectness[0].device)

for i in range(num_fmaps):
    H, W = objectness[i].shape[2], objectness[i].shape[3]
    batched_objectness_tensor[i,:,:(A * H * W)] = objectness[i].reshape(N, -1)
    batched_regression_tensor[i,:,:(4 * A * H * W)] = rpn_box_regression[i].reshape(N, -1)
    
batched_objectness_tensor = batched_objectness_tensor.reshape(num_fmaps * N, -1)
batched_objectness_tensor = batched_objectness_tensor.sigmoid()
batched_objectness_topk, topk_idx = batched_objectness_tensor.topk(2000, dim=1, sorted=True)

batched_anchor_data = [anchor_boxes, 
                       anchor_visibility, 
                       image_sizes_wh]

batched_anchor_tensor, image_shapes = batched_anchor_data[0], batched_anchor_data[2]

def test_nms():
    proposals_gen, objectness_gen, keep_gen = _C.GeneratePreNMSUprightBoxesBatched(
                                N,
                                A,
                                H_max*W_max,
                                A*H_max*W_max,
                                fmap_size_cat,
                                num_anchors_per_level,
                                topk_idx,
                                batched_objectness_topk.float(),    # Need to cast these as kernel doesn't support fp16
                                torch.rand([5, 8, 729600]).float(),
                                batched_anchor_tensor,
                                image_sizes.float(),
                                2000,
                                0,
                                4.135166556742356,
                                True)
    
if __name__=='__main__':
    test_nms()
