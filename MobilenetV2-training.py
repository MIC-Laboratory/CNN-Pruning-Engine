import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
from torchsummary import summary
from mobilenetv2 import MobileNetV2 as model
from torch.utils.tensorboard import SummaryWriter
import copy
import os

batch_size = 128
input_size = 32
fineTurningEpoch = range(210)
VGG_Layer_Number = 13
lr_rate = 0.1
momentum = 0.9
weight_decay = 5e-4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Data preparation
transform = transforms.Compose(
    [
    transforms.RandomCrop(32,4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])


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


block_channel_origin = [32,96,144,144,192,192,192,384,384,384,384,576,576,576,960,960,960]

net = model(config = block_channel_origin)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr_rate, momentum=momentum,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

best_acc = 0

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = lr_rate * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# training
def validation(epoch,network,file_name="VGG.pth"):
    
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

    

for epoch in fineTurningEpoch:
    # adjust_learning_rate(optimizer, epoch)
    train(epoch, network=net, optimizer=optimizer)
    validation(epoch, network=net,file_name="New_VGG.pth")
    scheduler.step()