# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn
from sagemakercv.layers.nhwc import Conv2d_NHWC, nhwc_to_nchw_transform, nchw_to_nhwc_transform, interpolate_nhwc
from sagemakercv.layers.nhwc import MaxPool2d_NHWC 
from sagemakercv.layers.nhwc import init
from sagemakercv.layers.conv_module import ConvModule
from sagemakercv import _C

class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None, nhwc=False
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        self.nhwc = nhwc
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)
            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1, nhwc=nhwc)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1, nhwc=nhwc)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            interpolate_func = F.interpolate if not self.nhwc else interpolate_nhwc
            inner_top_down = interpolate_func(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            # TODO use size instead of scale to make it robust to different sizes
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))
        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1], self.nhwc)
            results.extend(last_results)
        if self.nhwc:
            results_nchw = []
            for i, f in enumerate(results):
                results_nchw.append(nhwc_to_nchw_transform(f))
            return [tuple(results), tuple(results_nchw)]
        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x, nhwc):
        op = MaxPool2d_NHWC(1,2,0) if nhwc else nn.MaxPool2d(1,2,0)
        return [op(x)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels, nhwc):
        super(LastLevelP6P7, self).__init__()
        conv = conv2d_NHWC if nhwc else nn.Conv2d
        self.p6 = conv(in_channels, out_channels, 3, 2, 1)
        self.p7 = conv(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            init.kaiming_uniform_(module.weight, a=1, nhwc=nhwc)
            init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]

class DetectionBlock(torch.nn.Module):
    """Detection block in YOLO neck.
    Let out_channels = n, the DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
        1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n.
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn=True,
                 activation='leaky'):
        super(DetectionBlock, self).__init__()
        double_out_channels = out_channels * 2
        kwargs = {'bn': bn, 'activation': activation}
        self.conv1 = ConvModule(in_channels, out_channels, 1, **kwargs)
        self.conv2 = ConvModule(out_channels, double_out_channels, 3, padding=1, **kwargs)
        self.conv3 = ConvModule(double_out_channels, out_channels, 1, **kwargs)
        self.conv4 = ConvModule(out_channels, double_out_channels, 3, padding=1, **kwargs)
        self.conv5 = ConvModule(double_out_channels, out_channels, 1, **kwargs)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        out = self.conv5(tmp)
        return out

class YoloV3FPN(torch.nn.Module):
    """The neck of YOLOV3.
    It can be treated as a simplified version of FPN. It
    will take the result from Darknet backbone and do some upsampling and
    concatenation. It will finally output the detection result.
    Note:
        The input feats should be from top to bottom.
            i.e., from high-lvl to low-lvl
        But YOLOV3Neck will process them in reversed order.
            i.e., from bottom (high-lvl) to top (low-lvl)
    Args:
        num_scales (int): The number of scales / stages.
        in_channels (List[int]): The number of input channels per scale.
        out_channels (List[int]): The number of output channels  per scale.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Dictionary to construct and config norm
            layer. Default: dict(type='BN', requires_grad=True)
        act_cfg (dict, optional): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 bn=True,
                 activation='leaky'):
        super(YoloV3FPN, self).__init__()
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        kwargs = {'bn': bn, 'activation': activation}

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        self.detect1 = DetectionBlock(in_channels[0], out_channels[0], **kwargs)
        for i in range(1, self.num_scales):
            in_c, out_c = self.in_channels[i], self.out_channels[i]
            inter_c = out_channels[i - 1]
            self.add_module(f'conv{i}', ConvModule(inter_c, out_c, 1, **kwargs))
            # in_c + out_c : High-lvl feats will be cat with low-lvl feats
            self.add_module(f'detect{i+1}',
                            DetectionBlock(in_c + out_c, out_c, **kwargs))

    def forward(self, feats):
        assert len(feats) == self.num_scales

        # processed from bottom (high-lvl) to top (low-lvl)
        outs = []
        out = self.detect1(feats[-1])
        outs.append(out)

        for i, x in enumerate(reversed(feats[:-1])):
            conv = getattr(self, f'conv{i+1}')
            tmp = conv(out)

            # Cat with low-lvl feats
            tmp = F.interpolate(tmp, scale_factor=2)
            tmp = torch.cat((tmp, x), 1)

            detect = getattr(self, f'detect{i+2}')
            out = detect(tmp)
            outs.append(out)

        return tuple(outs)
