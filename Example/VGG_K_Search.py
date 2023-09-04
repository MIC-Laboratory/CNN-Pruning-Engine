import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import Linear

from testcase_base import testcase_base
from Weapon.WarmUpLR import WarmUpLR
from utils import frozen_layer,deFrozen_layer,compare_models
from copy import deepcopy
from thop import profile
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.getcwd()))
from Pruning_engine.pruning_engine import pruning_engine

class VGG_testcase(testcase_base):
    def __init__(self,config_file_path):
        super().__init__(config_file_path)
        self.total_layer = 13
        self.tool_net = deepcopy(self.net)
        
        if self.pruning_method[:1] == "K":
            layer_store = self.get_layer_store()
            self.pruner = pruning_engine(
                self.pruning_method,
                self.pruning_ratio,
                layer_store_private_variable=layer_store,
                list_k=self.list_k
                )
        
    def pruning(self,k,pruning_ratio_list):
        layers = self.net.features
        pruning_ratio_idx = 0
        
        for layer in range(len(layers)):
            
            if isinstance(layers[layer],Conv2d):
                layers[layer].k_value = k[pruning_ratio_idx]
                if layers[layer].in_channels == 3:
                    self.pruner.set_pruning_ratio(pruning_ratio_list[pruning_ratio_idx])
                    self.pruner.set_layer(layers[layer],main_layer=True)
                    
                    sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
                    layers[layer] = self.pruner.remove_filter_by_index(sorted_idx)
                else:
                    self.pruner.set_pruning_ratio(pruning_ratio_list[pruning_ratio_idx])
                    self.pruner.set_layer(layers[layer],main_layer=True)
                    layers[layer] = self.pruner.remove_conv_filter_kernel()
                pruning_ratio_idx+=1
            elif isinstance(layers[layer], BatchNorm2d):
                self.pruner.set_layer(layers[layer])
                sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
                layers[layer] = self.pruner.remove_Bn(sorted_idx)
        linear_layer = self.net.classifier[0]
        self.pruner.set_layer(linear_layer)
        sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
        self.net.classifier[0] = self.pruner.remove_kernel_by_index(sorted_idx=sorted_idx,linear=True)
        

    
    def get_layer_store(self):
        layers = self.net.features
        result = []
        for layer in range(len(layers)):
            if isinstance(layers[layer],Conv2d):
                result.append(layers[layer])
        return result
config_file_path = "Example/VGG_config.yaml"
k_max = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
k_max = [i//2 for i in k_max]

vgg_testcase = VGG_testcase(config_file_path)
# print(vgg_testcase.net)
testing_criterion = nn.CrossEntropyLoss()

print("==> Start to search for k:")
for max_k in range(len(k_max)):
    search_used_k = [1 for _ in k_max]
    pruning_ratio_list = [0 for _ in k_max]
    writer = SummaryWriter(log_dir=vgg_testcase.log_path+f"/K_Selection/layer{max_k}")
    for k in range(1,k_max[max_k]):
        
        vgg_testcase.net.load_state_dict(torch.load(vgg_testcase.pruning_config["Model"]["Pretrained_weight_path"]))
        vgg_testcase.net.to(vgg_testcase.device)
        search_used_k[max_k] = k
        pruning_ratio_list[max_k] = 0.1
        vgg_testcase.pruning(search_used_k,pruning_ratio_list)
        loss,accuracy = vgg_testcase.validation(criterion=testing_criterion,save=False)
        writer.add_scalar('ACC', accuracy, k)
