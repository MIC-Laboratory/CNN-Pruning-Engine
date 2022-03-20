'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    
}
class LinView(nn.Module):
    def __init__(self):
        super(LinView, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
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
        return input * self.weight.view(*self.size_mask)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.output_features is not None
        )
class VGG(nn.Module):
    def __init__(self, vgg_name, vgg_cfg_pruning=None,last_layer = 512):
        super(VGG, self).__init__()
        if vgg_cfg_pruning is not None:
            self.features = self._make_layers(vgg_cfg_pruning)
        else:
            self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(last_layer, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True),
                           ]
                in_channels = x
        return nn.Sequential(*layers)

    def flatten_model(self,old_net):
        """Removes nested modules. Only works for VGG."""
        from collections import OrderedDict
        module_list, counter, inserted_view = [], 0, False
        gate_counter = 0
        print("printing network")
        print(" Hard codded network in vgg_bn.py")
        for m_indx, module in enumerate(old_net.modules()):
            if not isinstance(module, (nn.Sequential, VGG)):
                print(m_indx, module)
                if isinstance(module, nn.Linear) and not inserted_view:
                    module_list.append(('flatten', LinView()))
                    inserted_view = True

                # features.0
                # classifier
                prefix = "features"

                if m_indx > 30:
                    prefix = "classifier"
                if m_indx == 32:
                    counter = 0

                # prefix = ""

                module_list.append((prefix + str(counter), module))

                if isinstance(module, nn.BatchNorm2d):
                    planes = module.num_features
                    gate = GateLayer(planes, planes, [1, -1, 1, 1])
                    module_list.append(('gate%d' % (gate_counter), gate))
                    print("gate ", counter, planes)
                    gate_counter += 1


                if isinstance(module, nn.BatchNorm1d):
                    planes = module.num_features
                    gate = GateLayer(planes, planes, [1, -1])
                    module_list.append(('gate%d' % (gate_counter), gate))
                    print("gate ", counter, planes)
                    gate_counter += 1


                counter += 1
        new_net = nn.Sequential(OrderedDict(module_list))
        return new_net
def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()