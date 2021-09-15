## Amazon SageMakerCV Utilities

Amazon SageMakerCV Utilities (SMCV-Utils) provides a set of highly optimized PyTorch functions for computer vision. Within computer vision, there are a set of common operations which are difficult to optimize on GPUs. Namely, these include:

- Anchor Generation
- Non-Max Suppression
- ROI Align
- ROI Pooling
- Bounding Box IOU
- Proposal Matching
- Channel Last Convolution Operations

SMCV-Utils provides implementations of these operations directly in CUDA, meaning they run significantly faster than Python. SMCV-Utils is fully integrated with the Amazon PyTorch Deeep Learning container (DLC) and works with Amazon SageMakerCV. SMCV-Utils is based on similar specialized Cuda functions written by Facebook and Nvidia for the Detectron2 model and MLPerf, respectively, but disaggregated from their larger models, such that they are more extensible to new and custom deep learning computer vision models.

## Documentation


### Anchor Generation

The anchor generator takes an image, feature map sizes, anchor sizes, and strides, and returns a set of level anchors and indicators if they are valid for the image.

```
from smcv_utils import _C

image_height = 1199
image_width = 800
feature_size = [200, 304]
stride = 4
base_anchors = torch.tensor([[-22., -10., 25., 13.], [-14., -14.,  17.,  17.], [-10., -22.,  13.,  25.]]).to('cuda')
straddle_thresh = 0

anchors, inds_inside = _C.anchor_generator(image_height, 
                                           image_width 
                                           feature_size, 
                                           base_anchors, 
                                           stride, 
                                           straddle_thresh)
```


### Non-Max Suppression

Non-max suppression takes a series of anchors, and reduces overlaps by selection the anchors with the highest probability of containing an image.

```
N = 8 # batch size
A = 3 # anchors per location
H_max = 304 # feature map height
W_max = 200 # feature map width
num_fmaps = 5 # number of feature maps

```


### ROI Align and ROI Pooling

### Proposal Matching

### Channel Last Convolution

## License

This project is licensed under the Apache-2.0 License.

