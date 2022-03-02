import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import copy
import os
import numpy as np
import math
import time
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
from torchsummary import summary
from Vgg import VGG as model
from sklearn.cluster import KMeans
from random import randrange
from torch.utils.tensorboard import SummaryWriter

batch_size = 128
input_size = 32
fineTurningEpoch = range(20)
VGG_Layer_Number = 13
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

classes = trainset.classes

# Netword preparation

VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
vgg_cfg_pruning = VGG16
net = model(vgg_name="VGG16")
new_net = model(vgg_name="VGG16",vgg_cfg_pruning=vgg_cfg_pruning,last_layer=vgg_cfg_pruning[-2])
new_net.to(device)
net.to(device)
net.load_state_dict(torch.load("weight/acc93.52%_VGG.pth"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9,weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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






device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")

best_acc = 0

# get gradient

# training
def validation(epoch,network,file_name="VGG.pth",save=True):
    
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
                if (save):
                    PATH = os.path.join(os.getcwd(),"weight")
                    if not os.path.isdir(PATH):
                        os.mkdir(PATH)
                    PATH = os.path.join(PATH,"acc"+str(accuracy)+"%_"+file_name)
                    torch.save(network.state_dict(), PATH)
                    print("Save: Acc "+str(best_acc))
def train(epoch,network,optimizer):
    # loop over the dataset multiple times
    running_loss = 0.0
    total = 0
    correct = 0
    
    network.to(device)
    network.train()
    with tqdm(total=len(trainloader)) as pbar:
        for i, data in enumerate(trainloader, 0):
            
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

# fineTurningSetUp()

# Start pruning

def VGG16Pruning():
    # backward gradient
    # Choose test images

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
            bias_zero_tensor = torch.zeros(1,device=device)
            for idx in sorted_idx:
                weight[idx.item()] = weight_zero_tensor
                bias[idx.item()] = bias_zero_tensor
            nonZeroRows_weight = torch.abs(weight).sum(dim=(1,2,3)) > 0
            
            weight = weight[nonZeroRows_weight]
            bias = bias[bias != 0]
            return weight,bias
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
    last_pruning_index = []
    out_channel = []
    pruning_index = []
    finish = False

    for index,m in enumerate(net.features,0):
        if isinstance(m, nn.Conv2d):
            
            out_channel = new_net.features[index].weight.data.shape[0]
            remove_filter = m.weight.data.shape[0] - out_channel
            num_filter = m.weight.data.size()[0]
            n_clusters = int((1/10)*num_filter)
            m_weight_vector = m.weight.data.view(num_filter,-1).cpu()
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(m_weight_vector)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            group = [[] for _ in range(n_clusters)] 
            for idx in range(num_filter):
                group[labels[idx]].append(idx)
            lock_group_index = []
            copy_group = copy.deepcopy(group)
            for filter_index_group in copy_group:
                if len(filter_index_group) == 1:
                    group.remove(filter_index_group)
            

            # The reminding item in group can be pruned by some crition
            pruning_index_group = []
            pruning_left_index_group = [[] for _ in range(len(group))] 
            total_left_filter = sum(len(filter_index_group) for filter_index_group in group)
            percentage_group = [int(100*(len(filter_index_group)/total_left_filter)) for filter_index_group in group]
            pruning_amount_group = [int(remove_filter*(percentage/100)) for percentage in percentage_group]
            for counter,filter_index_group in enumerate(group,0):
                filetr_index_group_temp = copy.deepcopy(filter_index_group)
                importance = torch.sum(torch.abs(m.weight.data[filter_index_group]), dim=(1, 2, 3))
                sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
                filetr_index_group_temp = [filetr_index_group_temp[index] for index in list(sorted_idx)]
                for sub_index in sorted_idx[len(sorted_idx)-pruning_amount_group[counter]:]:
                    if len(filetr_index_group_temp) == 1:
                        continue
                    pruning_index_group.append(filetr_index_group_temp.pop(filetr_index_group_temp.index(filter_index_group[sub_index])))
                for left_index in filetr_index_group_temp:
                    pruning_left_index_group[counter].append(left_index)
            # first one is the least important weight and the last one is the most important weight


            while (len(pruning_index_group) < remove_filter):
                pruning_amount = len(pruning_index_group)
                for left_index in pruning_left_index_group:
                    if (len(left_index) <= 1):
                        continue
                    if (len(pruning_index_group) >= remove_filter):
                        break
                    pruning_index_group.append(left_index.pop(0))
                if (pruning_amount >= len(pruning_index_group)):
                    raise ValueError('infinity loop')
                    
            pruning_index = torch.tensor(pruning_index_group).to(device)
            new_net.features[index].weight.data,new_net.features[index].bias.data = remove_filter_by_index(m.weight.data, pruning_index,bias=m.bias.data)
            new_net.features[index].weight.data = remove_kernel_by_index(new_net.features[index].weight.data,last_pruning_index)
        
        if isinstance(m, nn.BatchNorm2d):
            new_net.features[index].weight.data,new_net.features[index].bias.data,new_net.features[index].running_mean.data,new_net.features[index].running_var.data = remove_filter_by_index(m.weight.data, pruning_index,bias=m.bias.data,mean=m.running_mean.data,var=m.running_var.data)
            last_pruning_index = pruning_index
    
    for index,layer in enumerate(net.classifier,0):
        if isinstance(layer, nn.Linear):
            if not finish:
                in_channels = new_net.classifier[index].weight.data.shape[1]
                new_net.classifier[index].weight.data =  remove_kernel_by_index(layer.weight.data,last_pruning_index)
                new_net.classifier[index].bias.data =  layer.bias.data.clone()
                finish = True
            if finish:
                new_net.classifier[index].weight.data = layer.weight.data.clone()
                new_net.classifier[index].bias.data =  layer.bias.data.clone()
    print("After: ")
    validation(0,network=new_net,file_name="VGG_Prune.pth",save=False)

def UpdateNet(index,precentage):
    global new_net
    global optimizer
    global scheduler
    global net
    if VGG16[index] == "M":
        index+=1
    new_VGG16 = []
    for idx in range(len(VGG16)):
        
        if idx == index:
            if not (precentage < 0.00001):
                new_VGG16.append(round((precentage)*VGG16[idx]))
            else:
                new_VGG16.append(1)
        else:
            new_VGG16.append(VGG16[idx])
    print("Layer # :",index,"from ",VGG16[index],"to",new_VGG16[index])
    new_net = model(vgg_name="VGG16",vgg_cfg_pruning=new_VGG16,last_layer=new_VGG16[-2])
    net = model(vgg_name="VGG16")
    net.to(device)
    net.load_state_dict(torch.load("weight/acc93.52%_VGG.pth"))
    optimizer = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    train(0, net, optimizer)


for idx in range(VGG_Layer_Number):
    writer = SummaryWriter("VGG-K-Mean/L1norm/layers"+str(idx))
    precentage = 0
    UpdateNet(idx, 1-precentage)
    for _ in range(10):
        VGG16Pruning()
        writer.add_scalar('VGG-K-Mean-L1norm', best_acc, (precentage)*100)
        precentage += 0.1 
        UpdateNet(idx, (1-precentage))
        best_acc = 0
    writer.close()
    
