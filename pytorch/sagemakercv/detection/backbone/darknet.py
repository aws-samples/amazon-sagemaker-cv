# darknet-53 backbone with choice to output darknet results through fc layers or pass in three output for fpn and yolo

import torch
from torch import nn
from sagemakercv.detection import registry
from .fpn import YoloV3FPN
from collections import OrderedDict

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(OrderedDict([
        ("conv", 
            nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
        ("bn", nn.BatchNorm2d(out_num)),
        ("relu", nn.LeakyReLU()),
    ]))


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)
        torch.nn.init.kaiming_uniform_(self.layer1.conv.weight)
        torch.nn.init.kaiming_uniform_(self.layer2.conv.weight)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block):
        super(Darknet53, self).__init__()

        self.conv1 = conv_batch(3, 32)
        torch.nn.init.kaiming_uniform_(self.conv1.conv.weight)
        self.conv2 = conv_batch(32, 64, stride=2)
        torch.nn.init.kaiming_uniform_(self.conv2.conv.weight)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        torch.nn.init.kaiming_uniform_(self.conv3.conv.weight)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        torch.nn.init.kaiming_uniform_(self.conv4.conv.weight)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        torch.nn.init.kaiming_uniform_(self.conv5.conv.weight)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        torch.nn.init.kaiming_uniform_(self.conv6.conv.weight)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        output_1 = out
        out = self.conv5(out)
        out = self.residual_block4(out)
        output_2 = out
        out = self.conv6(out)
        out = self.residual_block5(out)
        output_3 = out
        
        return output_1, output_2, output_3

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)
    
class Darknet53Classifier(nn.Module):
    def __init__(self, block, num_classes=1000):
        super(Darknet53Classifier, self).__init__()
        self.num_classes = num_classes
        self.backbone = Darknet53(block)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)
    
    def forward(self, x):
        output_1, output_2, output_3 = self.backbone(x)
        out = self.global_avg_pool(output_3)
        out = out.view(-1, 1024)
        out = self.fc(out)
        return out
    
def darknet53(num_classes):
    return Darknet53Classifier(DarkResidualBlock, num_classes)

class DarknetFPNBackbone(torch.nn.Module):
    
    def __init__(self, block=DarkResidualBlock, 
                 num_scales=3, 
                 in_channels=[1024, 512, 256], 
                 out_channels=[512, 256, 128]
                ):
        super(DarknetFPNBackbone, self).__init__()
        self.backbone = Darknet53(block)
        self.fpn = YoloV3FPN(num_scales, in_channels, out_channels)
    
    def forward(self, inputs):
        feature_maps = self.backbone(inputs)
        return self.fpn(feature_maps)
    
    def load_darknet_weights(self, weights_path):
        import numpy as np
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)   # First five are header values
        header_info = header # Needed to write header when saving weights
        weights = np.fromfile(fp, dtype=np.float32)         # The rest are weights
        print ("total len weights = ", weights.shape)
        fp.close()

        ptr = 0
        all_dict = self.state_dict()
        all_keys = self.state_dict().keys()
        # print (all_keys)
        last_bn_weight = None
        last_conv = None
        
        # for i, (k, v) in enumerate(all_dict.items()):
        for i, (k, v) in enumerate(all_dict.items()):
            if 'fpn' in k:
                continue
            elif 'bn' in k:
                if 'weight' in k:
                    last_bn_weight = v
                elif 'bias' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    ptr += num_b
                    # weight
                    v = last_bn_weight
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    ptr += num_b
                    last_bn_weight = None
                elif 'running_mean' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    ptr += num_b
                elif 'running_var' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    ptr += num_b
                    # conv
                    v = last_conv
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    ptr += num_b
                    last_conv = None
                else:
                    continue
            elif 'conv.' in k:
                if 'weight' in k:
                    last_conv = v
                else:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    ptr += num_b
                    # conv
                    v = last_conv
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    ptr += num_b
                    last_conv = None
    
    
# @registry.BACKBONES.register("DARKNET53-FPN")
def build_darknet_backbone(cfg):
    return DarknetFPNBackbone(num_scales=len(cfg.MODEL.FPN.IN_CHANNELS),
                              in_channels=cfg.MODEL.FPN.IN_CHANNELS,
                              out_channels=cfg.MODEL.FPN.OUT_CHANNELS)
