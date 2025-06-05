import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights

class ResNet(nn.Module):
    def __init__(self, model, replace_stride_with_dilation):
        super(ResNet, self).__init__()
        if model == 'resnet152':
            model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2, replace_stride_with_dilation=replace_stride_with_dilation)
        elif model == 'resnet101':
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2, replace_stride_with_dilation=replace_stride_with_dilation)
        else:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, replace_stride_with_dilation=replace_stride_with_dilation)

        self.resnet = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

    def forward(self, x):
        high_level_featues = self.resnet(x)
        return high_level_featues
        