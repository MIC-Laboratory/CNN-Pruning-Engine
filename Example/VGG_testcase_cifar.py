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
                    sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
                    layers[layer] = self.pruner.remove_filter_by_index(sorted_idx)
                else:
                    self.pruner.set_pruning_ratio(self.pruning_ratio_list[pruning_ratio_idx])
                    self.pruner.set_layer(layers[layer],main_layer=True)
                    sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
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
        

    def retraining(self):
        training_criterion = nn.KLDivLoss()
        testing_criterion = nn.CrossEntropyLoss()

        print("==> Base validation acc:")
        loss,accuracy = self.validation(criterion=testing_criterion,save=False)

        print("==> Start pruning")
        optimizer = optim.SGD(self.net.parameters(), lr=self.lr_rate,momentum=self.momentum,weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.training_epoch)
        warmup_scheduler = WarmUpLR(optimizer, len(self.train_loader) * self.warmup_epoch)

        print("==> Start retraining")
        for epoch in range(self.training_epoch + self.warmup_epoch):
            self.train(
                epoch, 
                optimizer=optimizer,
                criterion=training_criterion,
                warmup_scheduler=warmup_scheduler
                )
            loss,accuracy = self.validation(
                criterion=testing_criterion,
                optimizer=optimizer,
                )
            if (epoch > self.warmup_epoch):
                scheduler.step()
            self.writer.add_scalar('Test/Loss', loss, epoch)
            self.writer.add_scalar('Test/ACC', accuracy, epoch)
        self.writer.close()
        print("==> Finish")

    def hook_function(self,tool_net,forward_hook,backward_hook):
        copy_tool_net = deepcopy(tool_net)
        layers = copy_tool_net.features
        for layer in range(len(layers)):
            if isinstance(layers[layer],Conv2d):
                layers[layer].register_forward_hook(forward_hook)
                layers[layer].register_forward_hook(backward_hook)
            
            
        return copy_tool_net
    def get_layer_store(self,net):
        layers = net.features
        result = []
        for layer in range(len(layers)):
            if isinstance(layers[layer],Conv2d):
                result.append(layers[layer])

        return result





testcase = VGG_testcase("Example/VGG_config.yaml")
# testcase.config_pruning()
# testcase.layerwise_pruning()
testcase.fullayer_pruning()
