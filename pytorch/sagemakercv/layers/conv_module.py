import torch
import torch.nn.functional as F

class Mish(torch.nn.Module):
    """ The MISH activation function (https://github.com/digantamisra98/Mish) """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ConvModule(torch.nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bn=False,
                 activation=None
                 ):
        super(ConvModule, self).__init__()
        self.conv = torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=not bn,
                    )
        if bn:
            self.bn = torch.nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5)
        if activation=='leaky':
            self.activation = torch.nn.LeakyReLU(0.1)
        elif activation=='mish':
            self.activtion = Mish()
        elif activation=='relu':
            self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x