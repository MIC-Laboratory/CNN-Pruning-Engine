import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from copy import deepcopy
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import Linear

from testcase_base import testcase_base
sys.path.append(os.path.join(os.getcwd()))
from Pruning_engine.pruning_engine import pruning_engine
class Mobilenet_testcase_cifar(testcase_base):
    def __init__(self,config_file_path):
        super().__init__(config_file_path)
        self.total_layer = 17
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
        layers = self.net.layers
        pruning_ratio_idx = 0
        for layer in range(len(layers)):
            self.pruner.set_pruning_ratio(self.pruning_ratio_list[pruning_ratio_idx])
            self.pruner.set_layer(layers[layer].conv2,main_layer=True)
            remove_filter_idx = self.pruner.get_remove_filter_idx()["current_layer"]
            layers[layer].conv2 = self.pruner.remove_filter_by_index(remove_filter_idx,group=True)
            
            self.pruner.set_layer(layers[layer].bn2)
            remove_filter_idx = self.pruner.get_remove_filter_idx()["current_layer"]
            layers[layer].bn2 = self.pruner.remove_Bn(remove_filter_idx=remove_filter_idx)

            self.pruner.set_layer(layers[layer].conv1)
            remove_filter_idx = self.pruner.get_remove_filter_idx()["current_layer"]
            layers[layer].conv1 = self.pruner.remove_filter_by_index(remove_filter_idx)

            self.pruner.set_layer(layers[layer].bn1)
            remove_filter_idx = self.pruner.get_remove_filter_idx()["current_layer"]
            layers[layer].bn1 = self.pruner.remove_Bn(remove_filter_idx=remove_filter_idx)
            
            self.pruner.set_layer(layers[layer].conv3)
            remove_filter_idx = self.pruner.get_remove_filter_idx()["current_layer"]
            layers[layer].conv3 = self.pruner.remove_kernel_by_index(remove_filter_idx)
            pruning_ratio_idx += 1

    
    def hook_function(self,tool_net,forward_hook,backward_hook):
        copy_tool_net = deepcopy(tool_net)
        layers = copy_tool_net.layers
        for layer in range(len(layers)):
            
            layers[layer].conv2.register_forward_hook(forward_hook)
            layers[layer].bn2.register_forward_hook(forward_hook)
            layers[layer].conv2.register_full_backward_hook(backward_hook)
            layers[layer].bn2.register_full_backward_hook(backward_hook)
            

        return copy_tool_net
    def get_layer_store(self,net):
        result = []
        layers = net.layers
        for layer in range(len(layers)):
            result.append(layers[layer].conv2)
            result.append(layers[layer].bn2)
            
            
                
        return result
config_file_path = "Example/Mobilenet_config.yaml"

mb_testcase = Mobilenet_testcase_cifar(config_file_path)
# testcase.config_pruning()
mb_testcase.layerwise_pruning()
# mb_testcase.fullayer_pruning()