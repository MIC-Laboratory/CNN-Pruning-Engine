import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
from torchsummary import summary
from Vgg import VGG as model
from torch.utils.tensorboard import SummaryWriter
from gradientDataSet import CIFAR10 as gradientSet
from ptflops import get_model_complexity_info
import copy
import os

weight = "weight/acc93.52%_VGG.pth"
batch_size = 200
input_size = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


# Netword preparation


VGG16 = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
[1,3,5,6,8,9,10,11,12]

vgg_cfg_pruning = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

pruning_index = [1,3,5,6,8,9,10,11,12]

net = model(vgg_name="VGG16")
net.to(device)
new_net = model(vgg_name="VGG16")
new_net.to(device)
net.load_state_dict(torch.load(weight))
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

best_acc = 0

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


# training
def validation(network,file_name="VGG.pth",save=True,dir_name="/weight"):
    
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
                    PATH = os.path.join(os.getcwd(),dir_name)
                    if not os.path.isdir(PATH):
                        os.mkdir(PATH)
                    PATH = os.path.join(PATH,"acc"+str(accuracy)+"%_"+file_name)
                    torch.save(network.state_dict(), PATH)
                    print("Save: Acc "+str(best_acc))




def VGG16Pruning():

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

    sorted_idx = None
    last_sorted_idx = []
    index = 0
    convIdx = 0
    global new_net
    global net

    
    for old,new in zip(net.modules(),new_net.modules()):
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
            if isinstance(old, nn.Conv2d):
                if (VGG16[index] == VGG16[index-1]):

                    
                    importance = torch.sum(torch.abs(old.weight.data.clone()),dim=(0,2,3))
                    sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
                    out_channels = vgg_cfg_pruning[index]
                    sorted_idx = sorted_idx[out_channels:]
                    convIdx+=1
                else:
                    sorted_idx = []

                new.weight.data,new.bias.data = remove_filter_by_index(old.weight.data, sorted_idx,bias=old.bias.data)
                new.weight.data = remove_kernel_by_index(new.weight.data,last_sorted_idx)
                index+=1
            if isinstance(old, nn.BatchNorm2d):
                new.weight.data,new.bias.data,new.running_mean.data,new.running_var.data = remove_filter_by_index(old.weight.data, sorted_idx,bias=old.bias.data,mean=old.running_mean.data,var=old.running_var.data)
                last_sorted_idx = sorted_idx
            
            if isinstance(old,nn.Linear):
                new.weight.data = remove_kernel_by_index(old.weight.data,last_sorted_idx,classifier=True)
                new.bias.data = old.bias.data.clone()
                last_sorted_idx = []

        
    
    print("After Pruning: ")
    validation(network=new_net,save=True,dir_name="test/")
    compare_models(net,new_net)

def UpdateNet(precentage):
    global new_net
    global net
    global vgg_cfg_pruning

    vgg_cfg_pruning = copy.deepcopy(VGG16)
    
    for idx in range(len(vgg_cfg_pruning)):
        
        if (VGG16[idx] == VGG16[idx-1]):
            if not (precentage < 0.00001):
                vgg_cfg_pruning[idx] = (round((precentage)*vgg_cfg_pruning[idx]))
            else:
                vgg_cfg_pruning[idx] = 1

    print("Pruning Rate:",str((1-precentage)*100))
    new_net = model(vgg_name="VGG16")
    new_net.to(device)
    net = model(vgg_name="VGG16")
    net.to(device)
    net.load_state_dict(torch.load(weight))
        



precentage = 0

for _ in range(10):
    # writer = SummaryWriter("VGG-Data/VGG-Gradient-test")
    
    UpdateNet(1-precentage)
    
    VGG16Pruning()

    # macs, params = get_model_complexity_info(new_net, (3, 32, 32), as_strings=True,
    #                                        print_per_layer_stat=False, verbose=True)
    # writer.add_scalar('ACC', best_acc, (precentage)*100)
    # writer.add_scalar('Params(M)', float(params.split(" ")[0]), (precentage)*100)
    # writer.add_scalar('MACs(G)', float(macs.split(" ")[0]), (precentage)*100)
    precentage += 0.1 

    best_acc = 0
    # writer.close()
