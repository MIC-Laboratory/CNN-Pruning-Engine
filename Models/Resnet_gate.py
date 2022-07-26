'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GateLayer(nn.Module):
    def __init__(self, input_features, output_features, size_mask):
        super(GateLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.size_mask = size_mask
        self.weight = nn.Parameter(torch.ones(output_features))

        # for simpler way to find these layers
        self.do_not_update = True

    def forward(self, input):
        return input*self.weight.view(*self.size_mask)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.output_features is not None
        )
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,gate=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.gate1 = GateLayer(planes,planes,[1, -1, 1, 1])
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.gate2 = GateLayer(planes,planes,[1, -1, 1, 1])
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.gate = gate

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.gate1(out)
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.gate2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        if self.gate is not None:
            out = self.gate(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, gate=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.gate1 = GateLayer(planes,planes,[1, -1, 1, 1])
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.gate2 = GateLayer(planes,planes,[1, -1, 1, 1])
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.gate = gate
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.gate1(out)
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.gate1(out)
        out = F.relu(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        if self.gate is not None:
            out = self.gate(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,skip_gate = True):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        gate = skip_gate
        self.gate = gate
        if gate:
            # self.gate_skip1 = GateLayer(64,64,[1, -1, 1, 1])
            self.gate_skip64 = GateLayer(64*4,64*4,[1, -1, 1, 1])
            self.gate_skip128 = GateLayer(128*4,128*4,[1, -1, 1, 1])
            self.gate_skip256 = GateLayer(256*4,256*4,[1, -1, 1, 1])
            self.gate_skip512 = GateLayer(512*4,512*4,[1, -1, 1, 1])
            if block == BasicBlock:
                self.gate_skip64 = GateLayer(64, 64, [1, -1, 1, 1])
                self.gate_skip128 = GateLayer(128, 128, [1, -1, 1, 1])
                self.gate_skip256 = GateLayer(256, 256, [1, -1, 1, 1])
                self.gate_skip512 = GateLayer(512, 512, [1, -1, 1, 1])
        else:
            self.gate_skip64 = None
            self.gate_skip128 = None
            self.gate_skip256 = None
            self.gate_skip512 = None
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, gate = self.gate_skip64)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, gate = self.gate_skip128)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, gate = self.gate_skip256)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, gate = self.gate_skip512)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, gate=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, gate=gate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes=num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3],num_classes=num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3],num_classes=num_classes)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3],num_classes=num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()