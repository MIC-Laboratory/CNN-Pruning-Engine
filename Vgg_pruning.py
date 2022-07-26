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
from sklearn.metrics.pairwise import pairwise_distances
from torchvision import transforms
from Models.Imagenet.Vgg import vgg16_bn
from tqdm import tqdm
from Models.Vgg import VGG as model
from torch.utils.tensorboard import SummaryWriter
from Dataloader.Gradient_data_set_cifar import taylor_cifar10,taylor_cifar100
from Dataloader.Gradient_data_set_imagenet import taylor_imagenet
from ptflops import get_model_complexity_info
from sklearn.cluster import KMeans
from gap_statistic import OptimalK
from sklearn.cluster import SpectralClustering as SpectralKMeans

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
parser.add_argument('--pruning_method', type=str,help="method: L1norm,Taylor,K-L1norm,K-Taylor,K-Distance")
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
log_path = f"Experiment_data/{args.dataset}/{args.pruning_method}/vgg16/{args.pruning_mode}"
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
if args.dataset == "Cifar10":
    train_set = torchvision.datasets.CIFAR10(dataset_path,train=True,transform=train_transform,download=True)
    test_set = torchvision.datasets.CIFAR10(dataset_path,train=False,transform=test_transform,download=True)
    taylor_set = taylor_cifar10(dataset_path,train=True,transform=train_transform,download=True,taylor_number=add_gradient_image_size)
    taylor_loader = torch.utils.data.DataLoader(taylor_set, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)
    # k_mean_number = [31,10,12,20,97,120,25,67,157,132,154,36,103]
elif args.dataset == "Cifar100":
    train_set = torchvision.datasets.CIFAR100(dataset_path,train=True,transform=train_transform,download=True)
    test_set = torchvision.datasets.CIFAR100(dataset_path,train=False,transform=test_transform,download=True)
    taylor_set = taylor_cifar100(dataset_path,train=True,transform=train_transform,download=True,taylor_number=add_gradient_image_size)
    taylor_loader = torch.utils.data.DataLoader(taylor_set, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)
    # k_mean_number = [14,5,12,42,15,9,83,139,92,134,223,209,82]
elif args.dataset == "Imagenet":
    train_set = torchvision.datasets.ImageFolder(f"{dataset_path}/train",transform=train_transform)
    test_set = torchvision.datasets.ImageFolder(f"{dataset_path}/val",transform=test_transform)
    taylor_set = taylor_imagenet(f"{dataset_path}/train",transform=train_transform,data_limit=add_gradient_image_size)
    taylor_loader = torch.utils.data.DataLoader(taylor_set, batch_size=batch_size//(2**3),
                                          shuffle=False, num_workers=num_workers)
    # k_mean_number = [10,1,16,15,17,69,31,70,28,52,13,241,241]
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

k_mean_number = [10,1,16,15,17,69,31,70,28,52,13,241,241]
classes = len(train_set.classes)
optimalK = OptimalK(n_jobs=16,parallel_backend='multiprocessing ')

# Netword preparation


VGG16 = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

vgg_cfg_pruning = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

pruning_rate = [0,0,0,0,0,0,0,0,0,0,0,0,0]

if args.dataset == "Imagenet":
    tool_net = vgg16_bn(pretrained=True)
    new_net = vgg16_bn(pretrained=True)
else:
    tool_net = model(vgg_name="VGG16",num_class=classes)
    new_net = model(vgg_name="VGG16",num_class=classes)
    tool_net.load_state_dict(torch.load(weight_path))
    new_net.load_state_dict(torch.load(weight_path))
tool_net.to(device)
new_net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=tool_net.parameters(), lr=lr_rate, momentum=momentum,weight_decay=weight_decay)
best_acc = 0
mask_number = 1e10
mean_feature_map = ["" for _ in range(len(vgg_cfg_pruning))]
mean_gradient = ["" for _ in range(len(vgg_cfg_pruning))]

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

def canopy(X, T1, T2, distance_metric='euclidean', filemap=None):
    canopies = dict()
    X1_dist = pairwise_distances(X, metric=distance_metric)
    canopy_points = set(range(X.shape[0]))
    while canopy_points:
        point = canopy_points.pop()
        i = len(canopies)
        canopies[i] = {"c":point, "points": list(np.where(X1_dist[point] < T2)[0])}
        canopy_points = canopy_points.difference(set(np.where(X1_dist[point] < T1)[0]))
    if filemap:
        for canopy_id in canopies.keys():
            canopy = canopies.pop(canopy_id)
            canopy2 = {"c":filemap[canopy['c']], "points":list()}
            for point in canopy['points']:
                canopy2["points"].append(filemap[point])
            canopies[canopy_id] = canopy2
    return canopies


def train(network,optimizer,dataloader_iter):
    
    
    network.train()
    inputs, labels = next(dataloader_iter)
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = network(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    
    

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


def remove_filter_by_index(weight,sorted_idx,bias=None,mean=None,var=None):       
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
        else:
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
    gradient_layer = len(vgg_cfg_pruning)-1
    def forward_hook(model, input, output):
        nonlocal feature_map_layer
        if (feature_map_layer >= len(vgg_cfg_pruning)):
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
                gradient_layer = len(vgg_cfg_pruning)-1
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
        
    for m in tool_net.modules():
        if (isinstance(m,nn.Conv2d)):
            m.register_forward_hook(forward_hook)
            m.register_forward_hook(backward_hook)
    
    with tqdm(total=len(taylor_loader)) as pbar:
        
        for _ in range(len(taylor_loader)):
            start = time.time()
            train(tool_net,optimizer,taylor_loader_iter)
            gc.collect()
            pbar.update()
            pbar.set_description_str(f"training time: {time.time()-start}")
        
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
def Taylor(index):
    conv_idx = 0
    if args.dataset == "Imagenet":
        tool_net = vgg16_bn(pretrained=True)
        
    else:
        tool_net = model("VGG16",num_class=classes)
        tool_net.load_state_dict(torch.load(weight_path))
    tool_net.to(device)
    if index == 0:
        Taylor_add_gradient()

    for i, tool in enumerate(tool_net.modules()):
        if isinstance(tool, nn.Conv2d):
            if (index == conv_idx):
                cam_grad = mean_gradient[index]*mean_feature_map[index]
                cam_grad = torch.abs(cam_grad)
                criteria_for_layer = cam_grad / (torch.linalg.norm(cam_grad) + 1e-8)
                
                importance = torch.sum(criteria_for_layer,dim=(1,2))
                sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
                return sorted_idx
            conv_idx+=1

def calculate_K(index): 
    # return K
    return k_mean_number[index]
def Kmean(weight,index,sort_index):
    out_channel = vgg_cfg_pruning[index]
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

def K_Distance(weight,index):
    
    return distance(weight,index)

def K_L1norm(weight,index):
    sort_index = L1norm(weight)
    return Kmean(weight,index,sort_index)
def K_Taylor(index):
    conv_idx = 0
    if args.dataset == "Imagenet":
        tool_net = vgg16_bn(pretrained=True)
    else:
        tool_net = model("VGG16",num_class=classes)
        tool_net.load_state_dict(torch.load(weight_path))
    tool_net.to(device)
    sort_index = Taylor(index)
    

    for i, tool in enumerate(tool_net.modules()):
        if isinstance(tool, nn.Conv2d):
            if (index == conv_idx):
                return Kmean(tool.weight.data,index,sort_index)
            conv_idx+=1
            

def VGG16Pruning(pruning_layer):
    sorted_idx = None
    last_sorted_idx = []
    index = 0
    skip = False
    global new_net
    global net
    for new in new_net.modules():
        # Test purpose, ignore that
        
        if False:
            if isinstance(old, nn.Conv2d):
                new.weight.data = old.weight.data.clone()
                new.bias.data = old.bias.data.clone()
            if isinstance(old, nn.BatchNorm2d):
                new.weight.data = old.weight.data.clone()
                new.bias.data = old.bias.data.clone()
                new.running_mean = old.running_mean.clone()
                new.running_var = old.running_var.clone()
            if isinstance(old, nn.Linear):
                new.weight.data = old.weight.data.clone()
                new.bias.data = old.bias.data.clone()

        # real pruning
        else:
            if isinstance(new, nn.Conv2d):
                if index - pruning_layer == 1:
                    new.weight.data = remove_kernel_by_index(new.weight.data.clone(),last_sorted_idx)
                    new.in_channels -= len(last_sorted_idx)
                if (pruning_layer == index):
                    out_channels = vgg_cfg_pruning[index]
                    if args.pruning_method == "L1norm":
                        sorted_idx = L1norm(new.weight.data.clone())
                        sorted_idx = sorted_idx[out_channels:]
                    elif args.pruning_method == "Taylor":
                        sorted_idx = Taylor(pruning_layer)
                        sorted_idx = sorted_idx[out_channels:]
                    elif args.pruning_method == "K-Taylor":
                        sorted_idx = K_Taylor(pruning_layer)
                    elif args.pruning_method == "K-L1norm":
                        sorted_idx = K_L1norm(new.weight.data.clone(),pruning_layer)
                    elif args.pruning_method == "K-Distance":
                        sorted_idx = K_Distance(new.weight.data.clone(),pruning_layer)
                        sorted_idx = sorted_idx[out_channels:]
                    if len(sorted_idx) != 0: 
                        new.weight.data,new.bias.data = remove_filter_by_index(new.weight.data.clone(), sorted_idx,bias=new.bias.data.clone())
                        new.out_channels -= len(sorted_idx)
                        skip = False
                    else:
                        skip = True
                else:
                    skip = True
                index+=1
            if isinstance(new, nn.BatchNorm2d):
                if not skip:
                    new.weight.data,new.bias.data,new.running_mean.data,new.running_var.data = remove_filter_by_index(new.weight.data.clone(), sorted_idx,bias=new.bias.data.clone(),mean=new.running_mean.data.clone(),var=new.running_var.data.clone())
                    last_sorted_idx = sorted_idx
                    new.num_features -= len(last_sorted_idx)
            if isinstance(new,nn.Linear):
                if (pruning_layer == 12):
                    if args.dataset == "Imagenet":
                        new.weight.data = new.weight.data.clone()
                    else:
                        new.weight.data = remove_kernel_by_index(new.weight.data.clone(),last_sorted_idx,linear=True)
                        new.in_features -= len(last_sorted_idx)
                    new.bias.data = new.bias.data.clone()
                        
                    last_sorted_idx = []

        
    
    print("Finish Pruning: ")
    

def UpdateNet(index,percentage,reload=True):
    global tool_net
    global vgg_cfg_pruning
    vgg_cfg_pruning = copy.deepcopy(VGG16)
    
    for idx in range(len(vgg_cfg_pruning)):
        if (idx == index):
            if not (percentage < 0.00001):
                vgg_cfg_pruning[idx] = (round((percentage)*vgg_cfg_pruning[idx]))
            else:
                vgg_cfg_pruning[idx] = 1
    print('Pruning from Layer',str(index),'Pruning Rate from',str(VGG16[index]),"to",str(vgg_cfg_pruning[index]))
    if reload:
        weight_reload()
def weight_reload():
    global new_net
    if args.dataset == "Imagenet":
        new_net = vgg16_bn(pretrained=True)
    else:
        new_net = model("VGG16",num_class=classes)
        new_net.load_state_dict(torch.load(weight_path))
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
        for index in range(len(pruning_rate)): 
            if args.dataset == "Imagenet" and index == len(pruning_rate)-1:
                break  
            UpdateNet(index,1-pruning_rate[index],reload=False)
            VGG16Pruning(index)
        
        validation(new_net,test_loader,args.weight_path+f"/full_layer_pruned_{str(100*round(percentage,1))}%",save=False)
        
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
        if args.dataset == "Imagenet" and layer == len(pruning_rate)-1:
            break  
        percentage = 0
        pruning_rate[layer] = 0 
        for idx in range(6):
            best_acc = 0
            writer = SummaryWriter(log_dir=log_path+f"/layer{layer}")
            pruning_rate[layer] = percentage
            UpdateNet(layer,1-pruning_rate[layer])
            VGG16Pruning(layer)
            validation(network=new_net,save=False,dataloader=test_loader,file_name=args.weight_path+f"/layer{layer}_pruned_{str(100*round(percentage,1))}%")
            with torch.cuda.device(0):
                macs, params = get_model_complexity_info(new_net, (3, input_size, input_size), as_strings=False,
                                            print_per_layer_stat=False, verbose=True)
                writer.add_scalar('ACC', best_acc, round((pruning_rate[layer]),1)*100)
                writer.add_scalar('Params(M)', round(params/1e6,2), round((pruning_rate[layer]),1)*100)
                writer.add_scalar('MACs(M)', round(macs/1e6,2), round((pruning_rate[layer]),1)*100)
            
            percentage+=0.1
    writer.close()
    
def bruth_force_calculate_k():
    global best_acc
    global K
    for layer in range(0,len(VGG16)):
        if args.dataset == "Imagenet" and layer == len(pruning_rate)-1:
            break  
        percentage = 0.1
        pruning_rate[layer] = 0 
        for _ in range(1):
            K=1
            for _ in range(int((1/2)*VGG16[layer])):
                best_acc = 0
                writer = SummaryWriter(log_dir=log_path+f"/layer{layer}/pruning{round((pruning_rate[layer]),1)*100}%")
                pruning_rate[layer] = percentage
                UpdateNet(layer,1-pruning_rate[layer])
                VGG16Pruning(layer)
                validation(network=new_net,save=False,dataloader=test_loader,file_name=args.weight_path+f"/layer{layer}_pruned_{str(100*round(percentage,1))}%")
                writer.add_scalar('ACC', best_acc, K)
                K+=1
            percentage+=0.1
    writer.close()


if args.pruning_mode == "Layerwise":
    layerwise_pruning()
elif args.pruning_mode == "Fullayer":
    full_layer_pruning()
# bruth_force_calculate_k()
