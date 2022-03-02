import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
from torchsummary import summary
# from mobilenetv2 import MobileNetV2 as model
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation

from mobilenetv2 import MobileNetV2 as models
from mobilenetv2 import Block as block

from gradientDataSet import CIFAR10 as gradientSet

import copy
import os

batch_size = 128
input_size = 32
fineTurningEpoch = range(200)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
best_acc = 0
lr_rate = 0.11
# Data preparation
transform = transforms.Compose(
    [
    transforms.CenterCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
gradient_Set = gradientSet(root='./data', train=True,download=True, transform=transform)

gradient_loader = torch.utils.data.DataLoader(gradient_Set, batch_size=batch_size,
                                         shuffle=True, num_workers=2)
classes = trainset.classes

block_channel_origin = [32,96,144,144,192,192,192,384,384,384,384,576,576,576,960,960,960]
block_channel_pruning = [32,96,144,144,192,192,192,384,384,384,384,576,576,576,960,960,960]

net = models(config = block_channel_origin)
net.to(device)
new_net = models(config = block_channel_pruning)
net.load_state_dict(torch.load("weight/acc93%_MobilenetV2.pth")["net"])



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr_rate, momentum=0.9,weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def validation(epoch,network,file_name="MobilenetV2_Prune.pth",save=True):
    
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

def train(epoch,network,optimizer,loader):
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

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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

    def remove_filter_by_index(weight,sorted_idx,bias=None,mean=None,var=None):
        
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
        else:
            weight_zero_tensor = torch.zeros(list(weight[0].size()),device=device)
            # bias_zero_tensor = torch.zeros(1,device=device)
            for idx in sorted_idx:
                weight[idx.item()] = weight_zero_tensor
                # bias[idx.item()] = bias_zero_tensor
            nonZeroRows_weight = torch.abs(weight).sum(dim=(1,2,3)) > 0
            
            weight = weight[nonZeroRows_weight]
            # bias = bias[bias != 0]
            # return weight,bias
            return weight
    def remove_kernel_by_index(weight,sorted_idx,classifier=None):
        weight_zero_tensor = torch.zeros(list(weight[0][0].size()),device=device)
        for idx in sorted_idx:
            weight[:,idx.item()] = weight_zero_tensor
        if (len(sorted_idx) != 0 and classifier == None):
            nonZeroRows_weight = torch.abs(weight).sum(dim=(0,2,3)) > 0 
            weight = weight[:,nonZeroRows_weight]
        if (classifier != None):
            weight = weight[:,weight[1]!=0]
        return weight
    
    sorted_idx = None
    valve = False
    index = -1
    skip_batch_norm = False
    shortcut_time = False
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
                        importance = torch.sum(torch.abs(old.weight.grad*old.weight.data), dim=(1, 2, 3))
                        # importance = torch.sum(torch.abs(old.weight.data), dim=(1, 2, 3))
                        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
                        pruning_amount = (old.weight.data.size(0) - new.weight.data.size(0))
                        total_size = len(sorted_idx)
                        sorted_idx = sorted_idx[total_size-pruning_amount:]
                        new.weight.data = remove_filter_by_index(old.weight.data, sorted_idx)
                    elif old.kernel_size != (1,1):
                        # new.weight.data = remove_kernel_by_index(old.weight.data.clone(), sorted_idx)
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
            
    print("After:")
    validation(0, new_net,save=False)
    
def UpdateNet(index,precentage):
    global new_net
    global optimizer
    global scheduler
    global net

    new_Mobilenet = []
    for idx in range(len(block_channel_origin)):
        
        if idx == index:
            if not (precentage < 0.00001):
                new_Mobilenet.append(round((precentage)*block_channel_origin[idx]))
            else:
                new_Mobilenet.append(1)
        else:
            new_Mobilenet.append(block_channel_origin[idx])
    print("Layer # :",index,"from ",block_channel_origin[index],"to",new_Mobilenet[index])
    new_net = models(config=new_Mobilenet)
    net = models(config=block_channel_origin)
    net.to(device)
    net.load_state_dict(torch.load("weight/acc93%_MobilenetV2.pth")["net"])
    
    optimizer = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9,weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    train(0, net, optimizer,gradient_loader)
    
    

for idx in range(len(block_channel_origin)):
    writer = SummaryWriter("MobilenetV2-data/MobilenetV2-Gradient/layer"+str(idx))
    precentage = 0
    UpdateNet(idx, 1-precentage)
    
    for _ in range(11):
        pruning()
        writer.add_scalar('Prune the smallest filters', best_acc, (precentage)*100)
        precentage += 0.1 
        UpdateNet(idx, (1-precentage))
        best_acc = 0
    writer.close()
