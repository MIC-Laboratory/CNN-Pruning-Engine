import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import copy
import os
import argparse
import gc
import time
import numpy as np
from torchvision import transforms
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from Dataloader.Gradient_data_set_cifar import taylor_cifar10,taylor_cifar100
from Dataloader.Gradient_data_set_imagenet import taylor_imagenet
from ptflops import get_model_complexity_info
from sklearn.cluster import KMeans
from Models.Imagenet.proxyless_nas.layers import ZeroLayer

parser = argparse.ArgumentParser(description="")
parser.add_argument('--dataset', type=str,help="dataset: Cifar10,Cifar100,Imagenet")
parser.add_argument('--dataset_path', type=str,help="Imagenet dataset path or Cifar10/Cifar100 datasetPath")
parser.add_argument('--pruning_mode', type=str,help="mode: Layerwise,Fullayer")
parser.add_argument('--pruning_method', type=str,help="method: L1norm,Taylor,K-L1norm,K-Taylor,K-Distance")
parser.add_argument('--calculate_k', type=str,help="options: Imagenet_K,Own_K ")

args = parser.parse_args()
print("==> Setting up hyper-parameters...")
batch_size = 128
training_epoch = 200
num_workers = 4
lr_rate = 0.01
momentum = 0.9
weight_decay = 5e-4
best_acc = 0
add_gradient_image_size = 100
dataset_mean = [0.485, 0.456, 0.406]
dataset_std = [0.229, 0.224, 0.225]
input_size = 224
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset_path = args.dataset_path
log_path = f"Experiment_data/ProxylessNAS/{args.calculate_k}/{args.dataset}/{args.pruning_method}/ProxylessNAS/{args.pruning_mode}"
train_transform = transforms.Compose(
    [
    transforms.CenterCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_mean,std=dataset_std)
    ])

test_transform = transforms.Compose([
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_mean,std=dataset_std)
])


# Data Preparation
print("==> Preparing data")
train_set = torchvision.datasets.ImageFolder(f"{dataset_path}/train",transform=train_transform)
test_set = torchvision.datasets.ImageFolder(f"{dataset_path}/val",transform=test_transform)
taylor_set = taylor_imagenet(f"{dataset_path}/train",transform=train_transform,data_limit=add_gradient_image_size)
taylor_loader = torch.utils.data.DataLoader(taylor_set, batch_size=batch_size//(2**3),
                                          shuffle=False, num_workers=num_workers)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
classes = len(train_set.classes)


# Netword preparation

block_channel_origin = [48,96,96,120,120,120,240,240,240,240,480,288,288,288,576,1152,576,576,1152]
block_channel_pruning = [48,96,96,120,120,120,240,240,240,240,480,288,288,288,576,1152,576,576,1152]

pruning_rate = [0 for _ in range(len(block_channel_origin))]
tool_net = torch.hub.load('mit-han-lab/ProxylessNAS', "proxyless_mobile", pretrained=True)
new_net = torch.hub.load('mit-han-lab/ProxylessNAS', "proxyless_mobile", pretrained=True)
tool_net.to(device)
new_net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=tool_net.parameters(), lr=lr_rate, momentum=momentum,weight_decay=weight_decay)
best_acc = 0
mask_number = 1e10
mean_feature_map = ["" for _ in range(len(block_channel_origin))]
mean_gradient = ["" for _ in range(len(block_channel_origin))]

"""
if args.calculate_k == "Imagenet_K":
    k_mean_number = [25,25,46,14,108,104,108,99]
elif args.calculate_k == "Cifar10_K":
    k_mean_number = [30,30,36,31,21,32,83,66]
elif args.calculate_k == "Cifar100_K":
    k_mean_number = [21,21,62,6,104,98,223,113]
"""
K = 1
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


def train(network,optimizer,dataloader_iter):
    
    
    network.train()
    inputs, labels = next(dataloader_iter)
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = network(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    

def validation(network,dataloader):
    
    # loop over the dataset multiple times
    global best_acc
    accuracy = 0
    running_loss = 0.0
    total = 0
    correct = 0
    network.eval()
        
    with tqdm(total=len(dataloader)) as pbar:
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                # forward + backward + optimize
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                running_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                pbar.update()
                pbar.set_description_str("Acc: {:.3f} {}/{} | Loss: {:.3f}".format(accuracy,correct,total,running_loss/(i+1)))
            
    
    return running_loss/len(dataloader),accuracy


def remove_filter_by_index(weight,sorted_idx,bias=None,mean=None,var=None,gate=False):       
        if mean is not None:
            mask_tensor = torch.tensor(mask_number,device=device)
            for idx in sorted_idx:
                weight[idx.item()] = mask_tensor
                bias[idx.item()] = mask_tensor
                mean[idx.item()] = mask_tensor 
                var[idx.item()] = mask_tensor
            weight = weight[weight != mask_number]
            bias = bias[bias != mask_number]
            mean = mean[mean != mask_number]
            var = var[var != mask_number]
            return weight,bias,mean,var
        elif gate:
            mask_tensor = torch.tensor(mask_number,device=device)
            for idx in sorted_idx:
                weight[idx.item()] = mask_tensor            
            weight = weight[weight != mask_number]
            return weight
        elif bias is not None:
            mask_tensor = torch.tensor(mask_number,device=device)
            mask_tensor = mask_tensor.repeat(list(weight[0].size()))
            bias_mask_tensor = torch.tensor(mask_number,device=device)
            for idx in sorted_idx:
                weight[idx.item()] = mask_tensor
                bias[idx.item()] = bias_mask_tensor
            nonMaskRows_weight = abs(torch.abs(weight).sum(dim=(1,2,3)) - torch.abs(mask_tensor).sum(dim=(0,1,2))) > mask_number
            weight = weight[nonMaskRows_weight]
            bias = bias[bias != mask_number]
            return weight,bias
        else:
            mask_tensor = torch.tensor(mask_number,device=device)
            mask_tensor = mask_tensor.repeat(list(weight[0].size()))
            for idx in sorted_idx:
                weight[idx.item()] = mask_tensor
            nonMaskRows_weight = abs(torch.abs(weight).sum(dim=(1,2,3)) - torch.abs(mask_tensor).sum(dim=(0,1,2))) > mask_number
            weight = weight[nonMaskRows_weight]
            return weight
def remove_kernel_by_index(weight,sorted_idx,linear=None):
    mask_tensor = torch.tensor(mask_number,device=device)
    mask_tensor = mask_tensor.repeat(list(weight[0][0].size()))
    for idx in sorted_idx:
        weight[:,idx.item()] = mask_tensor
    if (len(sorted_idx) != 0 and linear == None):
        nonMaskRows_weight = abs(torch.abs(weight).sum(dim=(2,3)) - torch.abs(mask_tensor).sum(dim=(0,1))) > 0.0001 
        weight = weight[:,nonMaskRows_weight[0]]
    if (linear != None):
        weight = weight[:,weight[1]!=mask_tensor]
    return weight
def L1norm(weight):
    importance = torch.sum(torch.abs(weight),dim=(1,2,3))
    sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
    return sorted_idx

def Taylor_add_gradient():
    global mean_feature_map
    global mean_gradient
    feature_map_layer = 0
    taylor_loader_iter = iter(taylor_loader)
    gradient_layer = len(block_channel_pruning)-1
    def forward_hook(model, input, output):
        nonlocal feature_map_layer
        if (feature_map_layer >= len(block_channel_pruning)):
            feature_map_layer = 0
        if mean_feature_map[feature_map_layer] == "":
            if len(taylor_loader) > 1:
                mean_feature_map[feature_map_layer] = torch.sum(output.detach(),dim=(0))/(add_gradient_image_size*classes)
            else:
                mean_feature_map[feature_map_layer] = output.detach()/(add_gradient_image_size*classes)
        else:
            if len(taylor_loader) > 1:
                mean_feature_map[feature_map_layer] = torch.add(mean_feature_map[feature_map_layer],torch.sum(output.detach(),dim=(0))/(add_gradient_image_size*classes))
            else:
                mean_feature_map[feature_map_layer] = torch.add(mean_feature_map[feature_map_layer],output.detach()/(add_gradient_image_size*classes))
        feature_map_layer+=1
    def backward_hook(model,input,output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return
        def _store_grad(grad):
            nonlocal gradient_layer
            if (gradient_layer < 0):
                gradient_layer = len(block_channel_pruning)-1
            if mean_gradient[gradient_layer] == '':
                if len(taylor_loader) > 1:
                    mean_gradient[gradient_layer] = torch.sum(grad.detach(),dim=(0))/(add_gradient_image_size*classes)
                else:
                    mean_gradient[gradient_layer] = grad.detach()/(add_gradient_image_size*classes)
            else:
                if len(taylor_loader) > 1:
                    mean_gradient[gradient_layer] = torch.add(mean_gradient[gradient_layer],torch.sum(grad.detach(),dim=(0))/(add_gradient_image_size*classes))
                else:
                    mean_gradient[gradient_layer] = torch.add(mean_gradient[gradient_layer],grad.detach()/(add_gradient_image_size*classes))
            gradient_layer-=1
        output.register_hook(_store_grad)
    for block in range(len(block_channel_origin)):
        layer = new_net.blocks[block+1].mobile_inverted_conv
        if isinstance(layer,ZeroLayer):
            continue
        layer.depth_conv.conv.register_forward_hook(forward_hook)
        layer.depth_conv.conv.register_forward_hook(backward_hook)
    
    with tqdm(total=len(taylor_loader)) as pbar:
        
        for _ in range(len(taylor_loader)):
            start = time.time()
            train(tool_net,optimizer,taylor_loader_iter)
            gc.collect()
            pbar.update()
            pbar.set_description_str(f"training time: {time.time()-start}")
def clear_mean_gradient_feature_map():
    global mean_gradient,mean_feature_map
    mean_gradient = ["" for _ in range(len(block_channel_origin))]
    mean_feature_map = ["" for _ in range(len(block_channel_origin))]
def Taylor(index):
    tool_net = torch.hub.load('mit-han-lab/ProxylessNAS', "proxyless_mobile", pretrained=True)
    tool_net.to(device)
    if index == 0:
        clear_mean_gradient_feature_map()
        Taylor_add_gradient()

    cam_grad = mean_gradient[index]*mean_feature_map[index]
    cam_grad = torch.abs(cam_grad)
    criteria_for_layer = cam_grad / (torch.linalg.norm(cam_grad) + 1e-8)
    
    importance = torch.sum(criteria_for_layer,dim=(1,2))
    sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
    return sorted_idx

def calculate_K(index): 
    # return k_mean_number[index]
    return K
def Kmean(weight,index,sort_index):
    out_channel = block_channel_pruning[index]
    remove_filter = weight.shape[0] - out_channel
    num_filter = weight.data.size()[0]

    
    n_clusters = calculate_K(index)
    m_weight_vector = weight.reshape(num_filter, -1).cpu()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(m_weight_vector)
    print("K:",n_clusters)
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
    total_left_filter = sum(len(filter_index_group)
                            for filter_index_group in group)
    percentage_group = [int(
        100*(len(filter_index_group)/total_left_filter)) for filter_index_group in group]
    pruning_amount_group = [
        int(remove_filter*(percentage/100)) for percentage in percentage_group]
    sorted_idx_origin = sort_index
    for counter, filter_index_group in enumerate(group, 0):
        temp = copy.deepcopy(filter_index_group)
        temp.sort(key=lambda e: (list(sorted_idx_origin).index(e),e) if e in list(sorted_idx_origin)  else (len(list(sorted_idx_origin)),e))
        sorted_idx = torch.tensor(temp,device=device)
        filetr_index_group_temp = copy.deepcopy(list(sorted_idx))
        
        for sub_index in sorted_idx[len(sorted_idx)-pruning_amount_group[counter]:]:
            if len(filetr_index_group_temp) == 1:
                continue
            pruning_index_group.append(filetr_index_group_temp.pop(filetr_index_group_temp.index(sub_index)))
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
            pruning_index_group.append(left_index.pop(-1))
        if (pruning_amount >= len(pruning_index_group)):
            raise ValueError('infinity loop')

    return torch.tensor(pruning_index_group).to(device)
def K_L1norm(weight,index):
    sort_index = L1norm(weight)
    return Kmean(weight,index,sort_index)

def K_Taylor(index):
    block_idx = -1
    valve = False
    tool_net = torch.hub.load('mit-han-lab/ProxylessNAS', "proxyless_mobile", pretrained=True)

    tool_net.to(device)
    tool_net.to(device)
    sort_index = Taylor(index)
    
    valve = False
    for i, tool in enumerate(tool_net.modules()):
        if isinstance(tool, nn.Conv2d) and valve:
            if (index == conv_idx):
                return Kmean(tool.weight.data,index,sort_index)
            conv_idx+=1
            valve = False
def distance(weight,index):
    n_clusters = calculate_K(index)
    num_filter = weight.data.size()[0]
    m_weight_vector = weight.reshape(num_filter, -1).cpu()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(m_weight_vector)
    distance_set = kmeans.fit_transform(m_weight_vector)

    num_filter_list = [i for i in range(num_filter)]
    distance = distance_set[num_filter_list,kmeans.labels_]
    unique, index,counts = np.unique(kmeans.labels_, return_counts=True,return_index=True)
    lock_group = index[counts==1]
    distance[lock_group] = 1e10
    distance = torch.from_numpy(distance)
    sorted_importance, sorted_idx = torch.sort(distance, dim=0, descending=True)
    
    return sorted_idx
def K_Distance(weight,index):
    
    return distance(weight,index)
def get_sorted_idx(weight,pruning_block):
    out_channels = block_channel_pruning[pruning_block]
    if args.pruning_method == "L1norm":
        sorted_idx = L1norm(weight)
        sorted_idx = sorted_idx[out_channels:]
    elif args.pruning_method == "Taylor":
        sorted_idx = Taylor(pruning_block)
        sorted_idx = sorted_idx[out_channels:]
    elif args.pruning_method == "K-Taylor":
        sorted_idx = K_Taylor(pruning_block)
    elif args.pruning_method == "K-L1norm":
        sorted_idx = K_L1norm(weight,pruning_block)
    elif args.pruning_method == "K-Distance":
        sorted_idx = K_Distance(weight,pruning_block)
        sorted_idx = sorted_idx[out_channels:]
    return sorted_idx
def remove_Bn(layer,sorted_idx):
    layer.weight.data,\
    layer.bias.data,\
    layer.running_mean.data,\
    layer.running_var.data = remove_filter_by_index(
        layer.weight.data.clone(), 
        sorted_idx,
        bias=layer.bias.data.clone(),
        mean=layer.running_mean.data.clone(),
        var=layer.running_var.data.clone()
        )
    layer.num_features -= len(sorted_idx)
def remove_conv_filter_kernel(layer,sorted_idx):
    layer.weight.data = remove_filter_by_index(layer.weight.data.clone(), sorted_idx)
    layer.weight.data = remove_kernel_by_index(layer.weight.data.clone(), sorted_idx)
    layer.out_channels -= len(sorted_idx)
    layer.in_channels -= len(sorted_idx)
    layer.groups -= len(sorted_idx)


def ProxylessNasPruning(pruning_block):
    global new_net
    global block_channel_pruning
    layer = new_net.blocks[pruning_block+1].mobile_inverted_conv
    if isinstance(layer,ZeroLayer):
        print("Skip Pruning: Zero Layer")
        return
    sorted_idx = get_sorted_idx(layer.depth_conv.conv.weight.data.clone(),pruning_block)
    layer.inverted_bottleneck.conv.weight.data = remove_filter_by_index(layer.inverted_bottleneck.conv.weight.data.clone(),sorted_idx)
    layer.depth_conv.conv.weight.data = remove_filter_by_index(layer.depth_conv.conv.weight.data.clone(),sorted_idx)
    layer.point_linear.conv.weight.data = remove_kernel_by_index(layer.point_linear.conv.weight.data.clone(),sorted_idx)
    remove_Bn(layer.inverted_bottleneck.bn,sorted_idx)
    remove_Bn(layer.depth_conv.bn,sorted_idx)
    
    layer.inverted_bottleneck.conv.out_channels -= len(sorted_idx)
    layer.depth_conv.conv.out_channels -= len(sorted_idx)
    layer.depth_conv.conv.in_channels -= len(sorted_idx)
    layer.depth_conv.conv.groups -= len(sorted_idx)
    layer.point_linear.conv.in_channels -= len(sorted_idx)
    print("Finish Pruning: ")
    

def UpdateNet(index,percentage,reload=True):
    global tool_net
    global block_channel_pruning
    block_channel_pruning = copy.deepcopy(block_channel_origin)
    
    for idx in range(len(block_channel_pruning)):
        if (idx == index):
            if not (percentage < 0.00001):
                block_channel_pruning[idx] = (round((percentage)*block_channel_pruning[idx]))
            else:
                block_channel_pruning[idx] = 1
    print('Pruning from Layer',str(index),'Pruning Rate from',str(block_channel_origin[index]),"to",str(block_channel_pruning[index]))
    if reload:
        weight_reload()
def weight_reload():
    global new_net
    new_net = torch.hub.load('mit-han-lab/ProxylessNAS', "proxyless_mobile", pretrained=True)
    new_net.to(device)
    
def full_layer_pruning():
    percentage = 0
    global best_acc
    writer = SummaryWriter(log_dir=log_path)
    for idx in range(6):
        best_acc = 0
        weight_reload()
        for element in range(len(pruning_rate)):
            pruning_rate[element] = percentage
        for index in range(len(block_channel_pruning)):   

            UpdateNet(index,1-pruning_rate[index],reload=False)
            ProxylessNasPruning(index)
        
        validation(new_net,test_loader,args.weight_path+f"/full_block_pruned_{str(100*round(percentage,1))}%",save=False)
        
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(new_net, (3, input_size, input_size), as_strings=False,
                                                print_per_layer_stat=False, verbose=True)

            writer.add_scalar('ACC', best_acc, round((percentage),1)*100)
            writer.add_scalar('Params(M)', round(params/1e6,2), round((percentage),1)*100)
            writer.add_scalar('MACs(M)', round(macs/1e6,2), round((percentage),1)*100)
            writer.close()
        percentage += 0.1
def layerwise_pruning():
    global best_acc
    for layer in range(len(pruning_rate)):
        if args.dataset == 'Imagenet' and layer == 0:
                continue
        percentage = 0
        pruning_rate[layer] = 0 
        for idx in range(10):
            best_acc = 0
            writer = SummaryWriter(log_dir=log_path+f"/block{layer}")
            pruning_rate[layer] = percentage
            UpdateNet(layer,1-pruning_rate[layer])
            ProxylessNasPruning(layer)
            validation(network=new_net,save=False,dataloader=test_loader,file_name=args.weight_path+f"/block{layer}_pruned_{str(100*round(percentage,1))}%")
            with torch.cuda.device(0):
                macs, params = get_model_complexity_info(new_net, (3, input_size, input_size), as_strings=True,
                                            print_per_layer_stat=False, verbose=True)
                writer.add_scalar('ACC', best_acc, round((pruning_rate[layer]),1)*100)
                writer.add_scalar('Params(M)', float(params.split(" ")[0]), round((pruning_rate[layer]),1)*100)
                writer.add_scalar('MACs(M)', float(macs.split(" ")[0]), round((pruning_rate[layer]),1)*100)
            
            percentage+=0.1
    writer.close()
    
def bruth_force_calculate_k():
    global best_acc
    global K
    for layer in range(len(pruning_rate)):
        percentage = 0.1
        pruning_rate[layer] = 0 
        for _ in range(1):
            K=1
            for _ in range(int((1/2)*block_channel_origin[layer])):
                best_acc = 0
                writer = SummaryWriter(log_dir=log_path+f"/K_Selection/block{layer}/pruning{round((pruning_rate[layer]),1)*100}%")
                pruning_rate[layer] = percentage
                UpdateNet(layer,1-pruning_rate[layer])
                ProxylessNasPruning(layer)
                validation(network=new_net,dataloader=test_loader)
                writer.add_scalar('ACC', best_acc, K)
                K+=1
            percentage+=0.1
# if args.pruning_mode == "Layerwise":
#     layerwise_pruning()
# elif args.pruning_mode == "Fullayer":
#     full_layer_pruning()
bruth_force_calculate_k()
