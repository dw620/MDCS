import os
from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
import config
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F
from attention import attention_layer

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3mb4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(inplanes, planes, stride=1):
    return nn.Conv2d(inplanes, planes, stride=stride, kernel_size=3, padding=1, bias=False)


# TDFF
class BasicBlock1(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(BasicBlock1, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(inplanes, 64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        identify = x
        identify = self.conv4(identify)
        identify = self.bn4(identify)
        identify = self.relu(identify)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identify
        out = self.relu(out)
        return out


class AtrousBasicBlock1(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(AtrousBasicBlock1, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(inplanes, 64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, outplanes, kernel_size=3, stride=1, padding=5, dilation=5)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        identify = x
        identify = self.conv4(identify)
        identify = self.bn4(identify)
        identify = self.relu(identify)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identify
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=21, scale=1, downsample=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.basiclayer1 = BasicBlock1(256, 64)
        self.basiclayer2 = BasicBlock1(64, 256)
        self.basiclayer3 = BasicBlock1(256, 64)
        self.basiclayer4 = BasicBlock1(64, 256)
        self.basiclayer5 = AtrousBasicBlock1(256, 64)
        self.basiclayer6 = AtrousBasicBlock1(64, 256)
        self.basiclayer7 = AtrousBasicBlock1(256, 64)
        self.basiclayer8 = AtrousBasicBlock1(64, 256)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # Top layer
        # self.toplayer = nn.Conv2d(2048,256, kernel_size=1, stride=1, padding=0)
        #
        # # Lateral layers
        # self.latlayer1 = nn.Conv2d(1024,256, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer3 = nn.Conv2d(256,256, kernel_size=1, stride=1, padding=0)
        #
        # # Smooth layers
        # self.smooth0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.toplayer_bn = nn.BatchNorm2d(256)
        self.toplayer_relu = nn.ReLU(inplace=True)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer1_bn = nn.BatchNorm2d(256)
        self.latlayer1_relu = nn.ReLU(inplace=True)

        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_bn = nn.BatchNorm2d(256)
        self.latlayer2_relu = nn.ReLU(inplace=True)

        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_bn = nn.BatchNorm2d(256)
        self.latlayer3_relu = nn.ReLU(inplace=True)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_bn = nn.BatchNorm2d(256)
        self.smooth1_relu = nn.ReLU(inplace=True)

        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_bn = nn.BatchNorm2d(256)
        self.smooth2_relu = nn.ReLU(inplace=True)

        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_bn = nn.BatchNorm2d(256)
        self.smooth3_relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        self.scale = scale
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.attention_p2top3 = attention_layer()
        self.attention_p3top4 = attention_layer()
        self.attention_p4top5 = attention_layer()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        h = x  # [32, 3, 224, 224]
        h = self.conv1(h)  # [32, 128, 112, 112]
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.maxpool(h)

        # c1 = h  # [32, 128, 112, 112]
        h = self.layer1(h)
        c2 = h  # [32, 256, 56, 56]
        h = self.layer2(h)
        c3 = h  # [32, 512, 28, 28]
        h = self.layer3(h)
        c4 = h  # [32, 1024, 14, 14]
        h = self.layer4(h)
        c5 = h  # [32, 2048, 7, 7]

        # Top-down

        p5 = self.toplayer(c5)  # [32, 256, 7, 7]
        p5 = self.toplayer_relu(self.toplayer_bn(p5))  # [32, 256, 7, 7]

        # p5 = self.basiclayer1(c5)
        # p5 = self.basiclayer2(p5)


        c4 = self.latlayer1(c4)
        c4 = self.latlayer1_relu(self.latlayer1_bn(c4))

        m4_1 = self.basiclayer3(c4)
        m4 = self.basiclayer4(m4_1)
        p4 = self._upsample_add(p5, m4)  # [32, 256, 14, 14]

        # p4 = self._upsample_add(p5, c4)  # [32, 256, 14, 14]
        p4 = self.smooth1(p4)
        p4 = self.smooth1_relu(self.smooth1_bn(p4))

        c3 = self.latlayer2(c3)
        c3 = self.latlayer2_relu(self.latlayer2_bn(c3))  # [32, 256, 28, 28]
        #
        m3_1 = self.basiclayer5(c3)
        m3 = self.basiclayer6(m3_1)
        c3_5 = self._upsample_add(p5, m3)
        c3_5 = self._upsample_add(p5, c3)  # [32, 256, 28, 28]
        p3 = self._upsample_add(p4, c3_5)  # [32, 256, 28, 28]
        # p3 = self._upsample_add(p4, m3)  # [32, 256, 28, 28]
        p3 = self.smooth2(p3)
        p3 = self.smooth2_relu(self.smooth2_bn(p3))

        c2 = self.latlayer3(c2)  # [32, 256, 56, 56]
        c2 = self.latlayer3_relu(self.latlayer3_bn(c2))

        m2_1 = self.basiclayer7(c2)  # [32, 64, 56, 56]
        m2 = self.basiclayer8(m2_1)  # [32, 256, 56, 56]
        c2_5 = self._upsample_add(p5, m2)  # [32, 256, 56, 56]
        c2_4 = self._upsample_add(p4, c2_5)  # [32, 256, 56, 56]
        p2 = self._upsample_add(p3, c2_4)  # [32, 256, 56, 56]
        # p2 = self._upsample_add(p3, m2)  # [32, 256, 56, 56]
        p2 = self.smooth3(p2)
        p2 = self.smooth3_relu(self.smooth3_bn(p2))

        p3 = self._upsample(p3, p2)  # [32, 256, 56, 56]
        p4 = self._upsample(p4, p2)  # [32, 256, 56, 56]
        p5 = self._upsample(p5, p2)  # [32, 256, 56, 56]

        # p5 = p5  # [32, 256, 56, 56]
        # p4 = self.attention_p2top3(p5, p4)  # [32, 256, 56, 56]
        # p3 = self.attention_p3top4(p4, p3)  # [32, 256, 56, 56]
        # p2 = self.attention_p4top5(p3, p2)  # [32, 256, 56, 56]

        p2 = p2  # [32, 256, 56, 56]
        p3 = self.attention_p2top3(p2, p3)  # [32, 256, 56, 56]
        p4 = self.attention_p3top4(p3, p4)  # [32, 256, 56, 56]
        p5 = self.attention_p4top5(p4, p5)  # [32, 256, 56, 56]

        out = torch.cat((p2, p3, p4, p5), 1)

        out = self.avg_pool(out)  # [32, 1024, 1, 1]
        out = torch.flatten(out, 1)
        # [32, 1024]
        return out

        # n1 = torch.cat((p2_1, p3_1), 1)  # [32, 512, 56, 56]
        # n2 = torch.cat((p4_1, p5_1), 1)  # [32, 512, 56, 56]

        # out = torch.cat((p2_1, p3_1, p4_1, p5_1), 1)  # [32, 1024, 56, 56]
        # # out = torch.cat((n1, n2), 1)  # [32, 1024, 56, 56]
        # out = self.avg_pool(out)  # [32, 1024, 1, 1]
        # out = torch.flatten(out, 1)  # [32, 1024]
        # return out

        # n1 = torch.cat((p2, p3), 1)  # [32, 512, 56, 56]
        # n2 = torch.cat((p3, p4), 1)  # [32, 512, 56, 56]
        # n3 = torch.cat((p4, p5), 1)  # [32, 512, 56, 56]
        #
        # a1 = n1  # [32, 512, 56, 56]
        # a2 = self.attention_p3top4(n1, n2)  # [32, 512, 56, 56]
        # a3 = self.attention_p4top5(n2, n3)  # [32, 512, 56, 56]
        #
        # m1 = torch.cat((a1, a2), 1)  # [32, 1024, 56, 56]
        # m2 = torch.cat((a2, a3), 1)  # [32, 1024, 56, 56]
        #
        # q1 = m1  # [32, 1024, 56, 56]
        # q2 = self.attention_p3top4(m1, m2)  # [32, 1024, 56, 56]
        #
        # t1 = torch.cat((q1, q2), 1)  # [32, 1024, 56, 56]
        #
        # out = self.avg_pool(t1)  # [32, 1024, 1, 1]
        # out = torch.flatten(out, 1)  # [32, 1024]
        # return out

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet50'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                if "fc" not in key and "features.13" not in key:
                    state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model


model = resnet50(pretrained=True, num_classes=1000)
print(model)



class Ranet(nn.Module):
    def __init__(self):
        super(Ranet, self).__init__()

        self.backbone = self._get_backbone()
        self.fc = nn.Linear(1024, config.NUM_CLASSES)
        # self.softmax = nn.LogSoftmax(dim=1)
    def _get_backbone(self):
        backbone = resnet50(pretrained=True, num_classes=1000)
        for param in backbone.layer1.parameters():
            param.requires_grad = False
        for param in backbone.layer2.parameters():
            param.requires_grad = False
        for param in backbone.layer3.parameters():
            param.requires_grad = False
        for param in backbone.layer4.parameters():
            param.requires_grad = False
        return backbone

    def forward(self, x):
        out = self.backbone(x)  # [32, 1024]
        out = out.view(-1, 1024)  # [32, 1024]
        out = self.fc(out)  # [32, 21]
        return out


