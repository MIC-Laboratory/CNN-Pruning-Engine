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
            layer_store = self.get_layer_store()
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
    def get_layer_store(self):
        layers = self.net.features
        result = []
        for layer in range(len(layers)):
            if isinstance(layers[layer],Conv2d):
                result.append(layers[layer])

        return result

# =========================================> Extra Experiment function

def VGG_pruning():
    config_file_path = "Example/VGG_config.yaml"
    vgg_testcase = VGG_testcase(config_file_path)

    print("Before pruning:",vgg_testcase.OpCounter())
    testing_criterion = nn.CrossEntropyLoss()

    print("==> Base validation acc:")
    loss,accuracy = vgg_testcase.validation(criterion=testing_criterion,save=False)
    # vgg_testcase.pruning()
    # vgg_testcase.retraining()
    loss,accuracy = vgg_testcase.validation(criterion=testing_criterion,save=False)
    print("After pruning:",vgg_testcase.OpCounter())

def VGG_layerwise_pruning():
    config_file_path = "Example/VGG_config.yaml"
    vgg_testcase = VGG_testcase(config_file_path)
    pruning_ratio_list_reference = deepcopy(vgg_testcase.pruning_ratio_list)
    vgg_testcase_net_reference = deepcopy(vgg_testcase.net)
    
    for layer_idx in range(len(pruning_ratio_list_reference)):
        vgg_testcase.pruning_ratio_list = deepcopy(pruning_ratio_list_reference)
        vgg_testcase.writer = SummaryWriter(log_dir=os.path.join(vgg_testcase.log_path,"Layer"+str(layer_idx)))
        accuracy_list = []
        mac_list = []
        for pruning_percentage in range(10):
            vgg_testcase.net = deepcopy(vgg_testcase_net_reference)
            vgg_testcase.pruning_ratio_list[layer_idx] = round(pruning_percentage/10,1)

            print("Pruning Ratio:",vgg_testcase.pruning_ratio_list[layer_idx])

            print("Before pruning:",vgg_testcase.OpCounter())
            testing_criterion = nn.CrossEntropyLoss()

            print("==> Base validation acc:")
            loss,accuracy = vgg_testcase.validation(criterion=testing_criterion,save=False)
            vgg_testcase.pruning()
            loss,accuracy = vgg_testcase.validation(criterion=testing_criterion,save=False)
            print("After pruning:",vgg_testcase.OpCounter())
            vgg_testcase.writer.add_scalar('Test/ACC', accuracy, pruning_percentage)
            accuracy_list.append(accuracy)
            mac_list.append(float(vgg_testcase.OpCounter()[0].rstrip('M')))
        
        # with open(vgg_testcase.log_path+"pruning_ratio.txt","a") as f:
        #     f.write(f"\n===============>Layer{layer_idx} Pruning Ratio==================>"+str(calculate_pruning_ratio(accuracy_list,mac_list,13)))
        vgg_testcase.writer.close()

def VGG_fullayer_pruning():
    config_file_path = "Example/VGG_config.yaml"
    vgg_testcase = VGG_testcase(config_file_path)
    vgg_testcase_net_reference = deepcopy(vgg_testcase.net)

    for pruning_percentage in range(20):
        for layer_idx in range(len(deepcopy(vgg_testcase.pruning_ratio_list))):
            vgg_testcase.pruning_ratio_list[layer_idx] = round(pruning_percentage/20,1)
        vgg_testcase.net = deepcopy(vgg_testcase_net_reference)
        print("Before pruning:",vgg_testcase.OpCounter())
        testing_criterion = nn.CrossEntropyLoss()

        print("==> Base validation acc:")
        loss,accuracy = vgg_testcase.validation(criterion=testing_criterion,save=False)
        vgg_testcase.pruning()
        loss,accuracy = vgg_testcase.validation(criterion=testing_criterion,save=False)
        print("After pruning:",vgg_testcase.OpCounter())
        vgg_testcase.writer.add_scalar('Test/ACC', accuracy, pruning_percentage)
    vgg_testcase.writer.close()

# =============== Formula: 
# For each layer, the average Mac decrease,
# divided by the average accuracy loss.
# to norm between 0 and 1, you will need first run
# with return raw_pruning_score to get the min and max value for raw_pruining score
# Then fit in to this formula: (value - min) / (max - min)
def calculate_pruning_ratio(accuracy_list,mac_list,num_conv_layer):

    accuracy_loss_list = []
    mac_loss_list = []

    for list_idx in range(len(accuracy_list)-1,0,-1):
        accuracy_loss_list.append(abs(accuracy_list[list_idx] - accuracy_list[list_idx-1]))
        mac_loss_list.append(abs(mac_list[list_idx] - mac_list[list_idx-1]))
    
    average_accuracy_loss = sum(accuracy_loss_list)/len(accuracy_loss_list)
    average_mac_loss = sum(mac_loss_list)/len(mac_loss_list)

    raw_pruning_score = round(average_mac_loss/average_accuracy_loss,2)
    # raw_pruning_score = round(average_accuracy_loss/average_mac_loss,2)
    return raw_pruning_score


# VGG_pruning()
# VGG_layerwise_pruning()
VGG_fullayer_pruning()
