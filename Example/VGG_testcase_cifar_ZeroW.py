import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import Linear

from testcase_base import testcase_base

from copy import deepcopy
from thop import profile
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.join(os.getcwd()))

from Pruning_engine.pruning_engine import pruning_engine

class VGG_testcase_cifar(testcase_base):
    def __init__(self,config_file_path):
        super().__init__(config_file_path)
        self.total_layer = 13
        self.tool_net = deepcopy(self.net)
        if self.pruning_method != "L1norm":
            layer_store = self.get_layer_store(self.net)
            self.pruner = pruning_engine(
                self.pruning_method,
                self.pruning_ratio,
                total_layer=self.total_layer, 
                taylor_loader=self.taylor_loader,
                total_sample_size=len(self.taylor_set), 
                hook_function=self.hook_function,
                tool_net=self.tool_net,
                layer_store_private_variable=layer_store,
                list_k=self.list_k
                )
        
    def pruning(self):
        layers = self.net.features
        pruning_ratio_idx = 0
        for layer in range(len(layers)):
            if isinstance(layers[layer],Conv2d):
                if layers[layer].in_channels == 3:
                    self.pruner.set_pruning_ratio(self.pruning_ratio_list[pruning_ratio_idx])
                    self.pruner.set_layer(layers[layer],main_layer=True)
                    remove_filter_idx = self.pruner.get_remove_filter_idx()["current_layer"]
                    layers[layer] = self.pruner.remove_filter_by_index(remove_filter_idx)
                else:
                    self.pruner.set_pruning_ratio(self.pruning_ratio_list[pruning_ratio_idx])
                    self.pruner.set_layer(layers[layer],main_layer=True)
                    remove_filter_idx = self.pruner.get_remove_filter_idx()["current_layer"]
                    layers[layer] = self.pruner.remove_conv_filter_kernel()
            
            elif isinstance(layers[layer], BatchNorm2d):
                self.pruner.set_layer(layers[layer])
                remove_filter_idx = self.pruner.get_remove_filter_idx()["current_layer"]
                layers[layer] = self.pruner.remove_Bn(remove_filter_idx)
        pruning_ratio_idx+=1
    def hook_function(self,tool_net,forward_hook,backward_hook):
        copy_tool_net = deepcopy(tool_net)
        layers = copy_tool_net.features
        for layer in range(len(layers)):
            if isinstance(layers[layer],Conv2d):
                layers[layer].register_forward_hook(forward_hook)
                layers[layer].register_full_backward_hook(backward_hook)
            if isinstance(layers[layer],BatchNorm2d):
                layers[layer].register_forward_hook(forward_hook)
                layers[layer].register_full_backward_hook(backward_hook)
            
        return copy_tool_net
    def get_layer_store(self,net):
        layers = net.features
        result = []
        for layer in range(len(layers)):
            if isinstance(layers[layer],Conv2d):
                result.append(layers[layer])
            if isinstance(layers[layer],BatchNorm2d):
                result.append(layers[layer])
        return result





testcase = VGG_testcase_cifar("Example/VGG_config.yaml")
# testcase.config_pruning()
# testcase.layerwise_pruning()
testcase.fullayer_pruning()
# testcase.k_search()
