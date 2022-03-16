'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
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

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, middle_channel, stride,gate=None):
        super(Block, self).__init__()
        self.stride = stride

        # planes = expansion * in_planes
        planes = middle_channel

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.gate1 = GateLayer(planes, planes, [1, -1, 1, 1])
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.gate2 = GateLayer(planes, planes, [1, -1, 1, 1])
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
        self.shortcut = nn.Sequential()
        self.gate = gate
        
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.gate1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.gate2(out)
        out = self.bn3(self.conv3(out))

        out = out + self.shortcut(x) if self.stride==1 else out
        # out = out + x if self.stride==1 else out
        if self.gate is not None:
            out = self.gate(out)
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)


    def __init__(self, num_classes=10,config = None):
        super(MobileNetV2, self).__init__()
        self.gate0 = GateLayer(16, 16, [1, -1, 1, 1])
        self.gate1 = GateLayer(24, 24, [1, -1, 1, 1])
        self.gate2 = GateLayer(32, 32, [1, -1, 1, 1])
        self.gate3 = GateLayer(64, 64, [1, -1, 1, 1])
        self.gate4 = GateLayer(96, 96, [1, -1, 1, 1])
        self.gate5 = GateLayer(160, 160, [1, -1, 1, 1])
        self.gate6 = GateLayer(320, 320, [1, -1, 1, 1])
        # middle channel, output channel, stride
        self.cfg = [
            (config[0],16,1,self.gate0),
            (config[1],24,1,self.gate1),
            (config[2],24,1,self.gate1),
            (config[3],32,2,self.gate2),
            (config[4],32,1,self.gate2),
            (config[5],32,1,self.gate2),
            (config[6],64,2,self.gate3),
            (config[7],64,1,self.gate3),
            (config[8],64,1,self.gate3),
            (config[9],64,1,self.gate3),
            (config[10],96,1,self.gate4),
            (config[11],96,1,self.gate4),
            (config[12],96,1,self.gate4),
            (config[13],160,2,self.gate5),
            (config[14],160,1,self.gate5),
            (config[15],160,1,self.gate5),
            (config[16],320,1,self.gate6)
            ]
        
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        
    def _make_layers(self, in_planes):
        layers = []


        for middle_channel, output_channel, stride, gate in self.cfg:
            if gate is not None:
                layers.append(Block(in_planes, output_channel, middle_channel, stride,gate))
                in_planes = output_channel
            else:
                layers.append(Block(in_planes, output_channel, middle_channel, stride))
                in_planes = output_channel
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
