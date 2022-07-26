import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import copy
import os
import argparse
import gc
from torchvision import transforms
from tqdm import tqdm
from Models.Mobilenetv2 import MobileNetV2 as cifar_mobilenet

from Models.Mobilenetv2 import Block as block
from Models.Mobilenetv2 import GateLayer as Gate
from torchvision.models import mobilenet_v2 as imagenet_mobilenet
from torchvision.models.mobilenetv2 import InvertedResidual as imagenet_mobilenet_block
from torch.utils.tensorboard import SummaryWriter
from Dataloader.Gradient_data_set_cifar import taylor_cifar10,taylor_cifar100
from Dataloader.Gradient_data_set_imagenet import taylor_imagenet
from ptflops import get_model_complexity_info
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser(description=
"""
This is training file. Training Resnet18, MobileNetV2 and VGG16 in Cifar10
argument: [models] [weight_path] [dataset]
ORDER MATTERS
"""
)
parser.add_argument('--weight_path', type=str)
parser.add_argument('--dataset', type=str,help="dataset: Cifar10,Cifar100,Imagenet")
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--pruning_mode', type=str,help="mode: Layerwise,Fullayer")
parser.add_argument('--pruning_method', type=str,help="method: L1norm,Taylor,K-L1norm,K-Taylor")
args = parser.parse_args()
print("==> Setting up hyper-parameters...")
batch_size = 128
training_epoch = 200
num_workers = 4
lr_rate = 0.01
momentum = 0.9
weight_decay = 5e-4
best_acc = 0
weight_path = args.weight_path
add_gradient_image_size = 100
if args.dataset == "Cifar10":
    dataset_mean = [0.4914, 0.4822, 0.4465]
    dataset_std = [0.2470, 0.2435, 0.2616]
    input_size = 32
elif args.dataset == "Cifar100":
    dataset_mean = [0.5071, 0.4867, 0.4408]
    dataset_std = [0.2675, 0.2565, 0.2761]
    input_size = 32
elif args.dataset == "Imagenet":
    dataset_mean = [0.485, 0.456, 0.406]
    dataset_std = [0.229, 0.224, 0.225]
    input_size = 224
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset_path = args.dataset_path
log_path = f"Experiment_data/{args.dataset}/{args.pruning_method}/mobilenetv2/{args.pruning_mode}"
train_transform = transforms.Compose(
    [
    transforms.RandomCrop(input_size),
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
if args.dataset == "Cifar10":
    train_set = torchvision.datasets.CIFAR10(dataset_path,train=True,transform=train_transform,download=True)
    test_set = torchvision.datasets.CIFAR10(dataset_path,train=False,transform=test_transform,download=True)
    taylor_set = taylor_cifar10(dataset_path,train=True,transform=train_transform,download=True,taylor_number=add_gradient_image_size)
elif args.dataset == "Cifar100":
    train_set = torchvision.datasets.CIFAR100(dataset_path,train=True,transform=train_transform,download=True)
    test_set = torchvision.datasets.CIFAR100(dataset_path,train=False,transform=test_transform,download=True)
    taylor_set = taylor_cifar100(dataset_path,train=True,transform=train_transform,download=True,taylor_number=add_gradient_image_size)
elif args.dataset == "Imagenet":
    train_set = torchvision.datasets.ImageFolder(f"{dataset_path}/train",transform=train_transform)
    test_set = torchvision.datasets.ImageFolder(f"{dataset_path}/val",transform=test_transform)
    taylor_set = taylor_imagenet(f"{dataset_path}/train",transform=train_transform,data_limit=add_gradient_image_size)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
taylor_loader = torch.utils.data.DataLoader(taylor_set,batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)
classes = len(train_set.classes)


# Netword preparation

block_channel_origin = [32,96,144,144,192,192,192,384,384,384,384,576,576,576,960,960,960]
block_channel_pruning = [32,96,144,144,192,192,192,384,384,384,384,576,576,576,960,960,960]

pruning_rate = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

if args.dataset == "Imagenet":
    tool_net = imagenet_mobilenet(pretrained=True)
    new_net = imagenet_mobilenet(pretrained=True)
else:
    tool_net = cifar_mobilenet(num_classes=classes)
    new_net = cifar_mobilenet(num_classes=classes)
    tool_net.load_state_dict(torch.load(weight_path))
    new_net.load_state_dict(torch.load(weight_path))
tool_net.to(device)
new_net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=tool_net.parameters(), lr=lr_rate, momentum=momentum,weight_decay=weight_decay)
best_acc = 0
mask_number = 1e10
mean_feature_map = ["" for _ in range(len(block_channel_origin))]
mean_gradient = ["" for _ in range(len(block_channel_origin))]
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


def train(epoch,network,optimizer,dataloader,limited_image = None):
    # loop over the dataset multiple times
    running_loss = 0.0
    total = 0
    correct = 0
    
    network.train()
    with tqdm(total=len(dataloader)) as pbar:
        for i, data in enumerate(dataloader, 0):
            if (limited_image is not None) and (i != limited_image):
                continue
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

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
            pbar.set_description_str("Epoch: {} | Acc: {:.3f} {}/{} | Loss: {:.3f}".format(epoch,accuracy,correct,total,running_loss/(i+1)))
            if (limited_image is not None):
                break
    

def validation(network,dataloader,file_name,save=True):
    
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
            if accuracy > best_acc:
                best_acc = accuracy
                if save:
                    if not os.path.isdir(weight_path):
                        os.mkdir(weight_path)
                    PATH = os.path.join(weight_path,"acc"+str(accuracy)+"%_"+file_name)
                    torch.save(network.state_dict(), PATH)
                    print("Save: Acc "+str(best_acc))
                else:
                    print("Best: Acc "+str(best_acc))
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
    gradient_layer = len(block_channel_pruning)-1
    def forward_hook(model, input, output):
        nonlocal feature_map_layer
        padding = add_gradient_image_size*classes - output.detach().size(0)
        if (feature_map_layer >= len(block_channel_pruning)):
            feature_map_layer = 0
        if mean_feature_map[feature_map_layer] == "":
            mean_feature_map[feature_map_layer] = torch.mean(torch.cat((output.detach(),torch.zeros(padding,output.detach().size(1),output.detach().size(2),output.detach().size(3),device=device)),0),dim=(0))
        else:
            mean_feature_map[feature_map_layer] = torch.add(mean_feature_map[feature_map_layer],torch.mean(output.detach(),dim=(0)))
        feature_map_layer+=1
    def backward_hook(model,input,output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return
        def _store_grad(grad):
            nonlocal gradient_layer
            padding = add_gradient_image_size*classes - grad.detach().size(0)
            if (gradient_layer < 0):
                gradient_layer = len(block_channel_pruning)-1
            if mean_gradient[gradient_layer] == '':
                mean_gradient[gradient_layer] = torch.mean(torch.cat((grad.detach(),torch.zeros(padding,grad.detach().size(1),grad.detach().size(2),grad.detach().size(3),device=device)),0),dim=(0))
            else:
                mean_gradient[gradient_layer] = torch.add(mean_gradient[gradient_layer],torch.mean(grad.detach(),dim=(0)))
            gradient_layer-=1
        output.register_hook(_store_grad)
    valve = False
    for m in tool_net.modules():
        if args.dataset == "Imagenet":
            if (isinstance(m,imagenet_mobilenet_block)):
                valve = True
        else:
            if (isinstance(m,block)):
                valve = True
        if (isinstance(m,nn.Conv2d)) and valve:
            m.register_forward_hook(forward_hook)
            m.register_forward_hook(backward_hook)
            valve = False
    for _ in range(len(taylor_loader)):
        gc.collect()
        train(0,tool_net,optimizer,taylor_loader,limited_image=_)

def Taylor(index):
    conv_idx = 0
    valve = False
    tool_net.load_state_dict(torch.load(weight_path))
    if index == 0:
        Taylor_add_gradient()

    for i, tool in enumerate(tool_net.modules()):
        if (isinstance(tool,block)):
            valve = True
        if isinstance(tool, nn.Conv2d) and valve:
            if (index == conv_idx):
                cam_grad = mean_gradient[index]*mean_feature_map[index]
                cam_grad = torch.abs(cam_grad)
                criteria_for_layer = cam_grad / (torch.linalg.norm(cam_grad) + 1e-8)
                
                importance = torch.sum(criteria_for_layer,dim=(1,2))
                sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
                return sorted_idx
            valve = False
            conv_idx+=1

def K_L1norm(weight,index):
    out_channel = block_channel_pruning[index]
    remove_filter = weight.shape[0] - out_channel
    num_filter = weight.data.size()[0]
    n_clusters = int((1/10)*num_filter)
    m_weight_vector = weight.reshape(num_filter, -1).cpu()
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
    total_left_filter = sum(len(filter_index_group)
                            for filter_index_group in group)
    percentage_group = [int(
        100*(len(filter_index_group)/total_left_filter)) for filter_index_group in group]
    pruning_amount_group = [
        int(remove_filter*(percentage/100)) for percentage in percentage_group]
    importance = torch.sum(
        torch.abs(weight), dim=(1, 2, 3))
    importance_1 = importance
    sorted_importance, sorted_idx_origin = torch.sort(
            importance_1, dim=0, descending=True)
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

def K_Taylor(index):
    block_idx = -1
    valve = False
    if args.dataset == "Imagenet":
        tool_net = imagenet_mobilenet(pretrained=True)
        
    else:
        tool_net = cifar_mobilenet(num_classes=classes)
        tool_net.load_state_dict(torch.load(weight_path))
    tool_net.to(device)
    if index == 0:
        Taylor_add_gradient()

    for i, tool in enumerate(tool_net.modules()):
        if isinstance(tool,block):
            block_idx+=1
            valve = True
        if isinstance(tool, nn.Conv2d) and valve:
            if (index == block_idx):
                out_channel = block_channel_pruning[index]

                remove_filter = tool.weight.data.shape[0] - out_channel
                num_filter = tool.weight.data.size()[0]
                n_clusters = int((1/10)*num_filter)
                m_weight_vector = tool.weight.data.reshape(num_filter,-1).cpu()
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
                cam_grad = mean_gradient[index]*mean_feature_map[index]
                cam_grad = torch.abs(cam_grad)
                criteria_for_layer = cam_grad / (torch.linalg.norm(cam_grad) + 1e-8)
                importance = torch.sum(criteria_for_layer,dim=(1,2))
                importance_1 = importance
                sorted_importance, sorted_idx_origin = torch.sort(
                        importance_1, dim=0, descending=True)
                for counter,filter_index_group in enumerate(group,0):
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
            
            valve = False
def MobilenetV2Pruning_Imagenet(pruning_block):
    sorted_idx = None
    valve = False
    index = -1
    skip_batch_norm = False
    convIndex = 0
    global new_net
    for new in new_net.modules():
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
            if index != pruning_block:
                valve = False
            if isinstance(new, imagenet_mobilenet_block):
                valve = True
                index+=1
                sorted_idx = None
            if valve:
                if isinstance(new, nn.Conv2d):
                    
                    if new.kernel_size == (1,1) and sorted_idx == None:

                        out_channels = block_channel_pruning[index]
                        if args.pruning_method == "L1norm":
                            sorted_idx = L1norm(new.weight.data.clone())
                            sorted_idx = sorted_idx[out_channels:]
                        elif args.pruning_method == "Taylor":
                            sorted_idx = Taylor(pruning_block)
                            sorted_idx = sorted_idx[out_channels:]
                        elif args.pruning_method == "K-Taylor":
                            sorted_idx = K_Taylor(pruning_block)
                        elif args.pruning_method == "K-L1norm":
                            sorted_idx = K_L1norm(new.weight.data.clone(),pruning_block)
                        
                        new.weight.data = remove_filter_by_index(new.weight.data.clone(), sorted_idx)
                        new.out_channels -= len(sorted_idx)
                        convIndex+=1
                    elif new.kernel_size != (1,1):
                        
                        new.weight.data = remove_filter_by_index(new.weight.data.clone(), sorted_idx)
                        new.groups = block_channel_pruning[index]
                        new.out_channels -= len(sorted_idx)
                    elif new.out_channels == 1280:
                        new.weight.data = new.weight.data.clone()
                        skip_batch_norm = True
                    elif index != 0:
                        new.weight.data = remove_kernel_by_index(new.weight.data.clone(), sorted_idx)
                        new.in_channels -= len(sorted_idx)
                        skip_batch_norm = True
                    
                if isinstance(new, nn.BatchNorm2d):
                    if (not skip_batch_norm):
                        new.weight.data,new.bias.data,new.running_mean,new.running_var = remove_filter_by_index(new.weight.data.clone(), sorted_idx,bias=new.bias.data,mean=new.running_mean,var=new.running_var)
                        new.num_features -= len(sorted_idx)
                    else:
                        new.weight.data = new.weight.data.clone()
                        new.bias.data = new.bias.data.clone()
                        new.running_mean = new.running_mean.clone()
                        new.running_var = new.running_var.clone()
                    skip_batch_norm = False
                
                
            else:
                if isinstance(new, nn.Conv2d):
                    new.weight.data = new.weight.data.clone()
                if isinstance(new, nn.BatchNorm2d):
                    new.weight.data = new.weight.data.clone()
                    new.bias.data = new.bias.data.clone()
                    new.running_mean = new.running_mean.clone()
                    new.running_var = new.running_var.clone()                        
                if isinstance(new, nn.Linear):
                    new.weight.data = new.weight.data.clone()
                    new.bias.data = new.bias.data.clone()


        
    
    print("Finish Pruning: ")

def MobilenetV2Pruning(pruning_block):
    sorted_idx = None
    valve = False
    index = -1
    skip_batch_norm = False
    shortcut_time = False
    convIndex = 0
    global new_net
    for new in new_net.modules():
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
            if index != pruning_block:
                valve = False
            if isinstance(new, block):
                valve = True
                index+=1
                sorted_idx = None
                shortcut_time = False
            if index >=0 and isinstance(new, nn.Sequential):
                shortcut_time = True
            if valve and not shortcut_time:
                if isinstance(new, nn.Conv2d):
                    
                    if new.kernel_size == (1,1) and sorted_idx == None:
                        out_channels = block_channel_pruning[index]
                        if args.pruning_method == "L1norm":
                            sorted_idx = L1norm(new.weight.data.clone())
                            sorted_idx = sorted_idx[out_channels:]
                        elif args.pruning_method == "Taylor":
                            sorted_idx = Taylor(pruning_block)
                            sorted_idx = sorted_idx[out_channels:]
                        elif args.pruning_method == "K-Taylor":
                            sorted_idx = K_Taylor(pruning_block)
                        elif args.pruning_method == "K-L1norm":
                            sorted_idx = K_L1norm(new.weight.data.clone(),pruning_block)
                        
                        new.weight.data = remove_filter_by_index(new.weight.data.clone(), sorted_idx)
                        new.out_channels -= len(sorted_idx)
                        convIndex+=1
                    elif new.kernel_size != (1,1):
                        
                        new.weight.data = remove_filter_by_index(new.weight.data.clone(), sorted_idx)
                        new.groups = block_channel_pruning[index]
                        new.out_channels -= len(sorted_idx)
                    else:
                        new.weight.data = remove_kernel_by_index(new.weight.data.clone(), sorted_idx)
                        new.in_channels -= len(sorted_idx)
                        skip_batch_norm = True
                if isinstance(new, nn.BatchNorm2d):
                    if (not skip_batch_norm):
                        new.weight.data,new.bias.data,new.running_mean,new.running_var = remove_filter_by_index(new.weight.data.clone(), sorted_idx,bias=new.bias.data,mean=new.running_mean,var=new.running_var)
                        new.num_features -= len(sorted_idx)
                    else:
                        new.weight.data = new.weight.data.clone()
                        new.bias.data = new.bias.data.clone()
                        new.running_mean = new.running_mean.clone()
                        new.running_var = new.running_var.clone()
                    skip_batch_norm = False
                if isinstance(new,Gate):
                    new.weight.data = remove_filter_by_index(new.weight.data.clone(), sorted_idx,gate=True)
                    new.output_features -= len(sorted_idx)
                
            else:
                if isinstance(new, nn.Conv2d):
                    new.weight.data = new.weight.data.clone()
                if isinstance(new, nn.BatchNorm2d):
                    new.weight.data = new.weight.data.clone()
                    new.bias.data = new.bias.data.clone()
                    new.running_mean = new.running_mean.clone()
                    new.running_var = new.running_var.clone()                        
                if isinstance(new, nn.Linear):
                    new.weight.data = new.weight.data.clone()
                    new.bias.data = new.bias.data.clone()
                if isinstance(new,Gate):
                    new.weight.data = new.weight.data.clone()

        
    
    print("Finish Pruning: ")
    

def UpdateNet(index,percentage,reload=True):
    global block_channel_pruning
    block_channel_pruning = copy.deepcopy(block_channel_origin)
    
    for idx in range(len(block_channel_pruning)):
        if (idx == index):
            if not (percentage < 0.00001):
                block_channel_pruning[idx] = (round((percentage)*block_channel_pruning[idx]))
            else:
                block_channel_pruning[idx] = 1
    print('Pruning from Block',str(index),'Pruning Rate from',str(block_channel_origin[index]),"to",str(block_channel_pruning[index]))
    if reload:
        weight_reload()
def weight_reload():
    global new_net
    if args.dataset == "Imagenet":
        new_net = imagenet_mobilenet(pretrained=True)
    else:
        new_net = cifar_mobilenet(num_classes=classes)
        new_net.load_state_dict(torch.load(weight_path))
    new_net.to(device)
    
def full_layer_pruning():
    percentage = 0
    global best_acc
    writer = SummaryWriter(log_dir=log_path)
    for idx in range(10):
        best_acc = 0
        weight_reload()
        for element in range(len(pruning_rate)):
            pruning_rate[element] = percentage
        for index in range(len(block_channel_pruning)):   
            if args.dataset == 'Imagenet' and index == 0:
                continue
            UpdateNet(index,1-pruning_rate[index],reload=False)
            if args.dataset == 'Imagenet':
                MobilenetV2Pruning_Imagenet(index)
            else:
                MobilenetV2Pruning(index)
        
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
            if args.dataset == 'Imagenet':
                MobilenetV2Pruning_Imagenet(layer)
            else:
                MobilenetV2Pruning(layer)
            validation(network=new_net,save=False,dataloader=test_loader,file_name=args.weight_path+f"/block{layer}_pruned_{str(100*round(percentage,1))}%")
            with torch.cuda.device(0):
                macs, params = get_model_complexity_info(new_net, (3, input_size, input_size), as_strings=True,
                                            print_per_layer_stat=False, verbose=True)
                writer.add_scalar('ACC', best_acc, round((pruning_rate[layer]),1)*100)
                writer.add_scalar('Params(M)', float(params.split(" ")[0]), round((pruning_rate[layer]),1)*100)
                writer.add_scalar('MACs(M)', float(macs.split(" ")[0]), round((pruning_rate[layer]),1)*100)
            
            percentage+=0.1
    writer.close()
    
if args.pruning_mode == "Layerwise":
    layerwise_pruning()
elif args.pruning_mode == "Fullayer":
    full_layer_pruning()

