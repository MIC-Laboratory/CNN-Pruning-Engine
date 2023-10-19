import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import Linear
from copy import deepcopy
from testcase_base import testcase_base
from Weapon.WarmUpLR import WarmUpLR
from utils import frozen_layer,deFrozen_layer,compare_models

sys.path.append(os.path.join(os.getcwd()))
from Pruning_engine.pruning_engine import pruning_engine

class ResNet_testcase(testcase_base):
    def __init__(self,config_file_path):
        super().__init__(config_file_path)
        self.total_layer = 33
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
        layersx = [
            self.net.layer1,
            self.net.layer2,
            self.net.layer3,
            self.net.layer4
        ]
        pruning_ratio_idx = 0
        for layers in layersx:
            for layer in range(len(layers)):
                self.pruner.set_pruning_ratio(self.pruning_ratio_list[pruning_ratio_idx])
                self.pruner.set_layer(layers[layer].conv2,main_layer=True)
                sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
                layers[layer].conv2 = self.pruner.remove_filter_by_index(sorted_idx)
                layers[layer].conv2 = self.pruner.remove_kernel_by_index(sorted_idx)
                

                self.pruner.set_layer(layers[layer].bn2)
                sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
                layers[layer].bn2 = self.pruner.remove_Bn(sorted_idx=sorted_idx)

                self.pruner.set_layer(layers[layer].conv1)
                sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
                layers[layer].conv1 = self.pruner.remove_filter_by_index(sorted_idx=sorted_idx)

                self.pruner.set_layer(layers[layer].bn1)
                sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
                layers[layer].bn1 = self.pruner.remove_Bn(sorted_idx=sorted_idx)
                
                self.pruner.set_layer(layers[layer].conv3)
                sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
                layers[layer].conv3 = self.pruner.remove_kernel_by_index(sorted_idx=sorted_idx)
                pruning_ratio_idx+=1
    


    def hook_function(self,tool_net,forward_hook,backward_hook):
        copy_tool_net = deepcopy(tool_net)
        layersx = [
            copy_tool_net.layer1,
            copy_tool_net.layer2,
            copy_tool_net.layer3,
            copy_tool_net.layer4
        ]
        for layers in layersx:
            for layer in range(len(layers)):
                layers[layer].conv2.register_forward_hook(forward_hook)
                layers[layer].conv2.register_forward_hook(backward_hook)

        return copy_tool_net
    def get_layer_store(self,net):
        result = []
        layersx = [
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4
        ]
        for layers in layersx:
            for layer in range(len(layers)):
                result.append(layers[layer].conv2)
                
        return result
config_file_path = "Example/ResNet_config.yaml"

testcase = ResNet_testcase(config_file_path)
# testcase.config_pruning()
# testcase.layerwise_pruning()
testcase.fullayer_pruning()
# testcase.k_search()