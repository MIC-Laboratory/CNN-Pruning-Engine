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
from Weapon.WarmUpLR import WarmUpLR
from utils import frozen_layer,deFrozen_layer,compare_models
sys.path.append(os.path.join(os.getcwd()))
from Pruning_engine.pruning_engine import pruning_engine
class Mobilenet_testcase(testcase_base):
    def __init__(self,config_file_path):
        super().__init__(config_file_path)
        self.total_layer = 17
        self.tool_net = deepcopy(self.net)
        if self.pruning_method == "Taylor":
            layer_store_grad = self.get_layer_store()
            self.pruner = pruning_engine(
                self.pruning_method,
                self.pruning_ratio,
                total_layer=self.total_layer, 
                taylor_loader=self.taylor_loader,
                total_sample_size=len(self.taylor_set), 
                hook_function=self.hook_function,
                tool_net=self.tool_net,
                layer_store_private_variable=get_layer_store
                )
        elif self.pruning_method[:1] == "K":
            layer_store = self.get_layer_store()
            self.pruner = pruning_engine(
                self.pruning_method,
                self.pruning_ratio,
                layer_store_private_variable=layer_store,
                list_k=self.list_k
                )
    
    def pruning(self):
        layers = self.net.layers
        for layer in range(len(layers)):
            self.pruner.set_layer(layers[layer].conv2,main_layer=True)
            sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
            layers[layer].conv2 = self.pruner.remove_filter_by_index(sorted_idx,group=True)
            
            self.pruner.set_layer(layers[layer].bn2)
            sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
            layers[layer].bn2 = self.pruner.remove_Bn(sorted_idx=sorted_idx)

            self.pruner.set_layer(layers[layer].conv1)
            sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
            layers[layer].conv1 = self.pruner.remove_filter_by_index(sorted_idx)

            self.pruner.set_layer(layers[layer].bn1)
            sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
            layers[layer].bn1 = self.pruner.remove_Bn(sorted_idx=sorted_idx)
            
            self.pruner.set_layer(layers[layer].conv3)
            sorted_idx = self.pruner.get_sorted_idx()["current_layer"]
            layers[layer].conv3 = self.pruner.remove_kernel_by_index(sorted_idx)
        

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
        layers = copy_tool_net.layers
        for layer in range(len(layers)):
            
            layers[layer].conv2.register_forward_hook(forward_hook)
            layers[layer].conv2.register_forward_hook(backward_hook)
            

        return copy_tool_net
    def get_layer_store(self):
        result = []
        layers = self.net.layers
        for layer in range(len(layers)):
            result.append(layers[layer].conv2)
            
            
                
        return result
config_file_path = "Example/Mobilenet_config.yaml"

mb_testcase = Mobilenet_testcase(config_file_path)
print(mb_testcase.net)
mb_testcase.pruning()
mb_testcase.retraining()