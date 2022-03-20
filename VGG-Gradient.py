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
from Vgg import GateLayer as gate
from torch.utils.tensorboard import SummaryWriter
from gradientDataSet import CIFAR10 as gradientSet
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from ptflops import get_model_complexity_info
import copy
import os

weight = "weight/acc93.52%_VGG.pth"
batch_size = 1024
input_size = 32
fineTurningEpoch = range(200)
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

gradient_Set = gradientSet(root='./data', train=True,download=True, transform=transform)

gradient_loader = torch.utils.data.DataLoader(gradient_Set, batch_size=batch_size,
                                         shuffle=True, num_workers=2)
classes = trainset.classes

# Netword preparation


VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


vgg_cfg_pruning = VGG16


net = model(vgg_name="VGG16",last_layer=512)
net.to(device)
# net = net.flatten_model(net)
new_net = model(vgg_name="VGG16",vgg_cfg_pruning=vgg_cfg_pruning,last_layer=vgg_cfg_pruning[-2])
# new_net = new_net.flatten_model(new_net)
net.load_state_dict(torch.load(weight))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9,weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

best_acc = 0
gradient = []
activation = None
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
            # optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()            
            
            
            accuracy = 100 * correct / total
            pbar.update()
            pbar.set_description_str("Loss: {:.3f} | Acc: {:.3f} {}/{}".format(running_loss/(i+1),accuracy,correct,total))


# Start pruning
def VGG16PruningSetUp():
    global activation
    
    for m in net.modules():

        def forward_hook(model, input, output):
            global activation
            activation.append(output.detach())

        if (isinstance(m,nn.Conv2d)):
            m.register_forward_hook(forward_hook)
        


def VGG16Pruning():

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
            # bias_zero_tensor = torch.zeros(1,device=device)
            for idx in sorted_idx:
                weight[idx.item()] = weight_zero_tensor
                # bias[idx.item()] = bias_zero_tensor
            nonZeroRows_weight = torch.abs(weight) > 0
            
            weight = weight[nonZeroRows_weight]
            # bias = bias[bias != 0]
            # return weight,bias
            return weight
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
            # return weight
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
    out_channel = []
    finish = False
    # global net
    for old,new in zip(net.modules(),new_net.modules()):
        # Test purpose, ignore that
        
        if False:
            if isinstance(m, nn.Conv2d):
                new_net.features[index].weight.data = m.weight.data.clone()
                new_net.features[index].bias.data = m.bias.data.clone()
            if isinstance(m, nn.BatchNorm2d):
                new_net.features[index].weight.data = m.weight.data.clone()
                new_net.features[index].bias.data = m.bias.data.clone()
                new_net.features[index].running_mean = m.running_mean.clone()
                new_net.features[index].running_var = m.running_var.clone()

        # real pruning
        else:
            if isinstance(old, nn.Conv2d):
                feature_map = torch.mean(activation[index],dim=(0,2,3))
                
                for x in range(feature_map.size(0)):
                    old.weight.grad[x,:,:,:]*=feature_map[x]

                old.weight.grad = torch.abs(old.weight.grad)
                # importance = torch.sum(m.weight.grad,dim=(0,2,3))
                
                criteria_for_layer = old.weight.grad / (torch.linalg.norm(old.weight.grad) + 1e-8)
                importance = torch.sum(criteria_for_layer,dim=(0,2,3))
                out_channels = new.weight.data.shape[0]
                

                sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
                sorted_idx = sorted_idx[out_channels:]
                new.weight.data,new.bias.data = remove_filter_by_index(old.weight.data, sorted_idx,bias=old.bias.data)
                new.weight.data = remove_kernel_by_index(new.weight.data,last_sorted_idx)
                index+=1
            if isinstance(old, nn.BatchNorm2d):
                new.weight.data,new.bias.data,new.running_mean.data,new.running_var.data = remove_filter_by_index(old.weight.data, sorted_idx,bias=old.bias.data,mean=old.running_mean.data,var=old.running_var.data)
                last_sorted_idx = sorted_idx

            if isinstance(old,gate):
                new.weight.data = remove_filter_by_index(old.weight.data, sorted_idx,gate=True)
            
            if isinstance(old,nn.Linear):
                new.weight.data = old.weight.data.clone()
                new.bias.data = old.bias.data.clone()

        
    
    print("After Pruning: ")
    validation(0,network=new_net,file_name="VGG_Gradient.pth",save=False)

def UpdateNet(index,precentage):
    global new_net
    global optimizer
    global scheduler
    global net
    global activation
    global gradient
    if VGG16[index] == "M":
        index+=1
    new_VGG16 = []
    for idx in range(len(VGG16)):
        if VGG16[idx] == "M":
            new_VGG16.append("M")
            continue
        if not (precentage < 0.00001):
            new_VGG16.append(round((precentage)*VGG16[idx]))
        else:
            new_VGG16.append(1)
    new_VGG16[-2]=512

        # if idx == index:
        #     if not (precentage < 0.00001):
        #         new_VGG16.append(round((precentage)*VGG16[idx]))
        #     else:
        #         new_VGG16.append(1)
        # else:
        #     new_VGG16.append(VGG16[idx])
    # print("Layer # :",index,"from ",VGG16[index],"to",new_VGG16[index])
    print("Pruning Rate:",str((1-precentage)*100))
    new_net = model(vgg_name="VGG16",vgg_cfg_pruning=new_VGG16,last_layer=new_VGG16[-2])
    net = model(vgg_name="VGG16")
    # net = net.flatten_model(net)
    net.to(device)
    # new_net = new_net.flatten_model(new_net)
    net.load_state_dict(torch.load(weight))
    
    
    optimizer = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9,weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    activation = []
    gradient = []
    
    VGG16PruningSetUp()
    train(0, net, optimizer,gradient_loader)

precentage = 0
writer = SummaryWriter("VGG-data/VGG-Gradient0")
for idx in range(10):
    
    
    UpdateNet(idx, 1-precentage)
    
    VGG16Pruning()
    macs, params = get_model_complexity_info(new_net, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
    writer.add_scalar('ACC', best_acc, (precentage)*100)
    writer.add_scalar('Params(M)', float(params.split(" ")[0]), (precentage)*100)
    writer.add_scalar('MACs(G)', float(macs.split(" ")[0]), (precentage)*100)
    precentage += 0.1 

    best_acc = 0
    writer.close()



# for idx in range(VGG_Layer_Number):
#     writer = SummaryWriter("VGG-Gradient/layer"+str(idx))
#     precentage = 0
#     UpdateNet(idx, 1-precentage)
    
#     for _ in range(11):
#         VGG16Pruning()
#         writer.add_scalar('Prune the smallest filters', best_acc, (precentage)*100)
#         precentage += 0.1 
#         UpdateNet(idx, (1-precentage))
#         best_acc = 0
#     writer.close()
# validation(0,network=net,file_name="VGG_Prune.pth",save=False)
