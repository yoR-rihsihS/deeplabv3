import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet
from .assp import ASPP

class DeepLabV3(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=20, output_stride=16):
        super(DeepLabV3, self).__init__()
        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            atrous_rates = [12, 24, 36]
        else:
            raise ValueError(f"output stride {output_stride} is not supported. Choose from '8' or '16'")
        
        if backbone in ['resnet50', 'resnet101', 'resnet152']:
            high_level_channels = 2048
        else:
            raise ValueError(f"backbone {backbone} is not supported. Choose from 'resnet50' or 'resnet101' or 'resnet152'")

        self.resnet = ResNet(model=backbone, replace_stride_with_dilation=replace_stride_with_dilation)
        self.aspp = ASPP(in_channels=high_level_channels, out_channels=256, atrous_rates=atrous_rates)
        self.classifier = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1, stride=1)

    def forward(self, image):
        size = image.shape[-2:]
        high_level_features = self.resnet(image)
        high_level_features = self.aspp(high_level_features)
        x = self.classifier(high_level_features)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x