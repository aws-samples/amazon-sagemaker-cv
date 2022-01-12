import torch
from smcv_utils import _C

mock_features = torch.rand([4, 256, 200, 304]).to('cuda')

mock_rois = torch.tensor([[0.0000e+00, 1.8608e+02, 1.4605e+02, 2.9531e+02, 2.5412e+02],
                            [0.0000e+00, 3.5388e+02, 1.5888e+02, 3.9574e+02, 2.7582e+02],
                            [1.0000e+00, 6.9182e+02, 3.7617e+02, 8.1049e+02, 4.4675e+02],
                            [1.0000e+00, 8.0659e+02, 6.1406e+02, 8.1942e+02, 6.3665e+02],
                            [1.0000e+00, 6.7051e+02, 4.1811e+02, 8.7968e+02, 5.1437e+02],
                            [2.0000e+00, 5.1053e+02, 7.0108e+01, 9.7426e+02, 7.8913e+02],
                            [2.0000e+00, 4.5144e+02, 2.1227e+01, 7.4031e+02, 6.1709e+02],
                            [3.0000e+00, 1.6321e+02, 5.0146e+02, 5.7366e+02, 7.4601e+02]]).to('cuda')

spatial_scale = .25
sampling_ratio = 2
output_size = (7, 7)
NHWC = False

def test_roi_align():
    extracted_features = _C.roi_align_forward(mock_features, mock_rois, spatial_scale, *output_size, sampling_ratio, NHWC)
    assert tuple(extracted_features.shape)==(8, 256, 7, 7)
    
if __name__=='__main__':
    test_roi_align()
