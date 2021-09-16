## Amazon SageMakerCV Utilities

Amazon SageMakerCV Utilities (SMCV-Utils) provides a set of highly optimized PyTorch functions for computer vision. Within computer vision, there are a set of common operations which are difficult to optimize on GPUs. Namely, these include:

- Anchor Generation
- Non-Max Suppression
- ROI Align
- Bounding Box IOU
- Proposal Matching
- Channel Last Convolution Operations

SMCV-Utils provides implementations of these operations directly in CUDA, meaning they run significantly faster than Python. SMCV-Utils is fully integrated with the Amazon PyTorch Deeep Learning container (DLC) and works with Amazon SageMakerCV. SMCV-Utils is based on similar specialized Cuda functions written by Facebook and Nvidia for the Detectron2 model and MLPerf, respectively, but disaggregated from their larger models, such that they are more extensible to new and custom deep learning computer vision models.

## Documentation


### Anchor Generation

The anchor generator takes an image, feature map sizes, anchor sizes, and strides, and returns a set of level anchors and indicators if they are valid for the image.

```
from smcv_utils import _C

anchors, inds_inside = _C.anchor_generator(image_height, 
                                           image_width 
                                           feature_size, # feature map dimensons list of 2 elements
                                           base_anchors, # tensor of size Nx4 specifying sizes of anchors at each level
                                           stride, # anchor strides along image
                                           straddle_thresh # stride starting position
                                           )
```


### Non-Max Suppression

Non-max suppression takes a series of anchors, and reduces overlaps by selection the anchors with the highest probability of containing an image.

```
from smcv_utils import _C

proposals, objectness, keep = _C.GeneratePreNMSUprightBoxes(
                                    N, # batch size
                                    A, # anchors per location
                                    H, # feature map height
                                    W, # feature map width
                                    topk_idx, # indices of top anchors
                                    objectness, # objectness output from RPN of size anchors
                                    box_regression_topk, # regression outputs of RPN 4xanchors
                                    anchors, # image base anchors
                                    image_shapes, # list batch size x 2
                                    nms_top_n, # number of regions to keep after nms
                                    min_size, # minimum regions
                                    bbox_xform_clip, # region clipping to feature maps
                                    return_indices # return keep indices
                                    )
```


### ROI Align

ROI align extracts sections from feature maps based on regions of interest.

```
extracted_features = _C.roi_align_forward(features, # single feature map (bs x width x height x channels) or (bs x channels x width x height)
                                          rois, # regions of interest (5 x num_rois) first column specifies element in batch
                                          spatial_scale, # size of feature map relative to image
                                          *output_size, # output feature map size
                                          sampling_ratio, # number of samples to take for bilinear interpolation
                                          NHWC # is data in channel last format
                                          )
```

### IOU

IOU computes the intersection over union of anchors versus targets

```
iou = _C.box_iou(torch.cat(rois, # (bs, 4, rois)
                           targets # (bs, 4, max number of boxes in batch)
                           )
```

### Proposal Matching

Matches ROIs to targets based on greatest IOU

```
matches = _C.match_proposals(iou, 
                             match_low_quality, # ensure at least one match per target (bool) 
                             low_overlap, # anchors below this are negative matches # float 
                             high_overlap # anchors above this are positive matches
                             )
```
### Channel Last Convolution

Channel last convolutions, along with max pooling and batch normalization, can be fastor on Nvidia GPUs than channel first, because of the order of memory accesses. These follow the same format as regular PyTorch convolutions, but expect data in a (bs, h, w, c) format.

```
import torch
from smcv_utils import NHWC

output =  NHWC.cudnn_convolution_nhwc(x, padded_w,
                                      padding, stride, dilation,
                                      groups,
                                      torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)


```


## License

This project is licensed under the Apache-2.0 License.

