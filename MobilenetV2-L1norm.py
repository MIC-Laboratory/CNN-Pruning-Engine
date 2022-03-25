from gc import garbage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation

from mobilenetv2 import MobileNetV2 as models

from mobilenetv2 import Block as block
from mobilenetv2 import GateLayer as Gate

from gradientDataSet import CIFAR10 as gradientSet
from ptflops import get_model_complexity_info

import copy
import os
import numpy as np

batch_size = 128
input_size = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
best_acc = 0
weight = "weight/acc94.38%_MobilenetV2.pth"
# Data preparation
transform = transforms.Compose(
    [
    transforms.CenterCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])




testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


block_channel_origin = [32,96,144,144,192,192,192,384,384,384,384,576,576,576,960,960,960]
block_channel_pruning = [32,96,144,144,192,192,192,384,384,384,384,576,576,576,960,960,960]

net = models(config = block_channel_origin)
net.to(device)
new_net = models(config = block_channel_pruning)
net.load_state_dict(torch.load(weight))
# 


criterion = nn.CrossEntropyLoss()



def validation(network,file_name="MobilenetV2_Prune.pth",save=True):
    
    # loop over the dataset multiple times
    total = 0
    correct = 0
    global best_acc
    network.to(device)
    network.eval()
    with tqdm(total=len(testloader)) as pbar:
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)


                # forward + backward + optimize
                outputs = network(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                pbar.update()
                pbar.set_description_str("Acc: {:.3f} {}/{}".format(accuracy,correct,total))
            if accuracy > best_acc:
                best_acc = accuracy
                if save:
                    PATH = os.path.join(os.getcwd(),"weight")
                    if not os.path.isdir(PATH):
                        os.mkdir(PATH)
                    PATH = os.path.join(PATH,"acc"+str(accuracy)+"%_"+file_name)
                    torch.save(network.state_dict(), PATH)
                    print("Save: Acc "+str(best_acc))

def train(network,loader):
    # loop over the dataset multiple times
    running_loss = 0.0
    total = 0
    correct = 0
    
    network.to(device)
    network.train()
    with tqdm(total=len(loader)) as pbar:
    
        for i, data in enumerate(loader, 0):
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)


            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()            
            
            
            accuracy = 100 * correct / total
            pbar.update()
            pbar.set_description_str("Loss: {:.3f} | Acc: {:.3f} {}/{}".format(running_loss/(i+1),accuracy,correct,total))
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


def pruning():

    def remove_filter_by_index(weight,sorted_idx,bias=None,mean=None,var=None,gate=False):
        
        if mean is not None:
            zero_tensor = torch.zeros(1,device=device)
            for idx in sorted_idx:
                weight[idx.item()] = zero_tensor
                bias[idx.item()] = zero_tensor
                mean[idx.item()] = zero_tensor 
                var[idx.item()] = zero_tensor
            weight = weight[weight != 0]
            bias = bias[bias != 0]
            mean = mean[mean != 0]
            var = var[var != 0]
            return weight,bias,mean,var
        elif gate:
            weight_zero_tensor = torch.zeros(1,device=device)
            for idx in sorted_idx:
                weight[idx.item()] = weight_zero_tensor
            nonZeroRows_weight = torch.abs(weight) > 0
            
            weight = weight[nonZeroRows_weight]

            return weight
        else:
            weight_zero_tensor = torch.zeros(list(weight[0].size()),device=device)
            for idx in sorted_idx:
                weight[idx.item()] = weight_zero_tensor
            nonZeroRows_weight = torch.abs(weight).sum(dim=(1,2,3)) > 0
            
            weight = weight[nonZeroRows_weight]

            return weight
    def remove_kernel_by_index(weight,sorted_idx):
        weight_zero_tensor = torch.zeros(list(weight[0][0].size()),device=device)
        for idx in sorted_idx:
            weight[:,idx.item()] = weight_zero_tensor
        if (len(sorted_idx) != 0):
            nonZeroRows_weight = torch.abs(weight).sum(dim=(0,2,3)) > 0 
            weight = weight[:,nonZeroRows_weight]

        return weight
    
    sorted_idx = None
    valve = False
    index = -1
    skip_batch_norm = False
    shortcut_time = False
    convIndex = 0
    global new_net
    for old,new in zip(net.modules(),new_net.modules()):
        if False:
            if isinstance(old, nn.Conv2d):
                new.weight.data = old.weight.data.clone()
            if isinstance(old, nn.BatchNorm2d):
                new.weight.data = old.weight.data.clone()
                new.bias.data = old.bias.data.clone()
                new.running_mean = old.running_mean.clone()
                new.running_var = old.running_var.clone()
            if isinstance(old, nn.Linear):
                new.weight.data = old.weight.data.clone()
                new.bias.data = old.bias.data.clone()
            if isinstance(old,Gate):
                new.weight.data = old.weight.data.clone()
        else:
            if index >= 17:
                valve = False
            if isinstance(old, block):
                valve = True
                index+=1
                sorted_idx = None
                shortcut_time = False
            if index >=0 and isinstance(old, nn.Sequential):
                shortcut_time = True
            if valve and not shortcut_time:
                if isinstance(old, nn.Conv2d):
                    if old.kernel_size == (1,1) and sorted_idx == None:
                        
                        
                        importance = torch.sum(torch.abs(old.weight.data), dim=(1, 2, 3))
                        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
                        
                        pruning_amount = (old.weight.data.size(0) - new.weight.data.size(0))
                        total_size = len(sorted_idx)
                        sorted_idx = sorted_idx[total_size-pruning_amount:]
                        new.weight.data = remove_filter_by_index(old.weight.data, sorted_idx)
                        
                        convIndex+=1
                    elif old.kernel_size != (1,1):
                        
                        new.weight.data = remove_filter_by_index(old.weight.data, sorted_idx)
                    else:
                        new.weight.data = remove_kernel_by_index(old.weight.data, sorted_idx)
                        skip_batch_norm = True
                if isinstance(old, nn.BatchNorm2d):
                    if (not skip_batch_norm):
                        new.weight.data,new.bias.data,new.running_mean,new.running_var = remove_filter_by_index( old.weight.data, sorted_idx,bias=old.bias.data,mean=old.running_mean,var=old.running_var)
                    else:
                        new.weight.data = old.weight.data.clone()
                        new.bias.data = old.bias.data.clone()
                        new.running_mean = old.running_mean.clone()
                        new.running_var = old.running_var.clone()
                    skip_batch_norm = False
                if isinstance(old,Gate):
                    new.weight.data = remove_filter_by_index(old.weight.data, sorted_idx,gate=True)
                
            else:
                if isinstance(old, nn.Conv2d):
                    new.weight.data = old.weight.data.clone()
                if isinstance(old, nn.BatchNorm2d):
                    new.weight.data = old.weight.data.clone()
                    new.bias.data = old.bias.data.clone()
                    new.running_mean = old.running_mean.clone()
                    new.running_var = old.running_var.clone()                        
                if isinstance(old, nn.Linear):
                    new.weight.data = old.weight.data.clone()
                    new.bias.data = old.bias.data.clone()
                if isinstance(old,Gate):
                    new.weight.data = old.weight.data.clone()
            
    print("Validation:")
    validation(new_net,save=False)
    
def UpdateNet(precentage):
    global new_net
    global net

    new_Mobilenet = []
    for idx in range(len(block_channel_origin)):
        if idx == 3 or idx == 6 or idx == 16:
            new_Mobilenet.append(block_channel_origin[idx])
            continue
        if not (precentage < 0.00001):
            new_Mobilenet.append(round((precentage)*block_channel_origin[idx]))
        else:
            new_Mobilenet.append(1)
    print("Pruning",str((1-precentage)*100)+"%")
    
    new_net = models(config=new_Mobilenet)
    net = models(config=block_channel_origin)
    net.to(device)
    net.load_state_dict(torch.load(weight))


 


precentage = 0
for idx in range(11):
    # writer = SummaryWriter("MobilenetV2-data/MobilenetV2-L1norm-no-gate")
    UpdateNet( 1-precentage)
    pruning()
    macs, params = get_model_complexity_info(new_net, (3, 32, 32), as_strings=True,
                                        print_per_layer_stat=False, verbose=True)
    # writer.add_scalar('ACC', best_acc, (precentage)*100)
    # writer.add_scalar('Params(M)', float(params.split(" ")[0]), (precentage)*100)
    # writer.add_scalar('MACs(G)', float(macs.split(" ")[0]), (precentage)*100)
    precentage += 0.1 
    best_acc = 0
    # writer.close()

