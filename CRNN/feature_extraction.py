import torch.nn as nn
from config import INPUT_CHANNEL, OUTPUT_CHANNEL

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

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

class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()
        # initial conv layers
        self.inplanes = output_channel // 8
        self.conv0_1 = nn.Conv2d(input_channel, output_channel // 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(output_channel // 16)
        self.conv0_2 = nn.Conv2d(output_channel // 16, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, output_channel // 4, layers[0])
        self.conv1   = nn.Conv2d(output_channel // 4, output_channel // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(output_channel // 4)

        # stage 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, output_channel // 2, layers[1], stride=1)
        self.conv2   = nn.Conv2d(output_channel // 2, output_channel // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2     = nn.BatchNorm2d(output_channel // 2)

        # stage 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2,1), padding=(0,1))
        self.layer3 = self._make_layer(block, output_channel, layers[2], stride=1)
        self.conv3   = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3     = nn.BatchNorm2d(output_channel)

        # stage 4
        self.layer4   = self._make_layer(block, output_channel, layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(output_channel, output_channel, kernel_size=2, stride=(2,1), padding=(0,1), bias=False)
        self.bn4_1   = nn.BatchNorm2d(output_channel)
        self.conv4_2 = nn.Conv2d(output_channel, output_channel, kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2   = nn.BatchNorm2d(output_channel)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        # matches ClovaAI's [1,2,5,3]
        self.ConvNet = ResNet(INPUT_CHANNEL, OUTPUT_CHANNEL, BasicBlock, [1,2,5,3])

    def forward(self, x):
        return self.ConvNet(x)
