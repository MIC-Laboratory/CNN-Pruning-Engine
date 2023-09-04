import sys
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import yaml
import torchvision
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy
from tqdm import tqdm
from utils import frozen_layer,deFrozen_layer,compare_models,seed_worker
from thop import profile,clever_format

sys.path.append(os.path.join(os.getcwd()))
from Models.Resnet import ResNet101
from Models.Mobilenetv2 import MobileNetV2
from Models.Vgg import VGG
from Pruning_engine.pruning_engine import pruning_engine
from Pruning_criterion.Taylor.Taylor_set_cifar import taylor_cifar10,taylor_cifar100
from Pruning_criterion.Taylor.Taylor_set_imagenet import taylor_imagenet
class testcase_base:

    def __init__(self,config_file_path,**kwargs):
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(42)
        with open(config_file_path,"r") as f:
            config = yaml.load(f,yaml.FullLoader)
            training_config = config["Training_seting"]
            pruning_config = config["Functionality_pruning_experiment"]
            self.pruning_config = pruning_config
        batch_size =  training_config["batch_size"]

        self.training_epoch =  training_config["training_epoch"]
        self.lr_rate =  training_config["learning_rate"]
        self.warmup_epoch =  training_config["warmup_epoch"]
        self.momentum = training_config["momentum"]
        self.weight_decay = training_config["weight_decay"]
        self.model_name = training_config["model"]
        num_workers =  training_config["num_workers"]
        best_acc = 0
        if  training_config["dataset"] == "Cifar10":
            dataset_mean = [0.4914, 0.4822, 0.4465]
            dataset_std = [0.2470, 0.2435, 0.2616]
            self.input_size = 32
        elif  training_config["dataset"] == "Cifar100":
            dataset_mean = [0.5071, 0.4867, 0.4408]
            dataset_std = [0.2675, 0.2565, 0.2761]
            self.input_size = 32
        elif  training_config["dataset"] == "Imagenet":
            dataset_mean = [0.485, 0.456, 0.406]
            dataset_std = [0.229, 0.224, 0.225]
            self.input_size = 224
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dataset_path =  training_config["dataset_path"]
        self.weight_path = os.path.join(
            training_config["weight_path"],
            training_config["dataset"],
            training_config["model"],
            pruning_config["Pruning"]["Pruning_mode"],
            pruning_config["Pruning"]["K_calculation"][1],
            pruning_config["Pruning"]["Pruning_method"]
            )
        log_path = os.path.join(
            training_config["experiment_data_path"],
            training_config["dataset"],
            training_config["model"],
            pruning_config["Pruning"]["Pruning_mode"],
            pruning_config["Pruning"]["K_calculation"][1],
            pruning_config["Pruning"]["Pruning_method"],
            ) 
        self.log_path = log_path
        self.writer = SummaryWriter(log_dir=log_path)
        train_transform = transforms.Compose(
            [
            transforms.RandomCrop(self.input_size,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.autoaugment.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            transforms.Normalize(mean=dataset_mean,std=dataset_std)
            ])

        test_transform = transforms.Compose([
            transforms.RandomCrop(self.input_size,padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean,std=dataset_std)
        ])


        # Data Preparation
        print("==> Preparing data")
        if  training_config["dataset"] == "Cifar10":
            self.train_set = torchvision.datasets.CIFAR10(dataset_path,train=True,transform=train_transform,download=True)
            self.test_set = torchvision.datasets.CIFAR10(dataset_path,train=False,transform=test_transform,download=True)
            self.taylor_set = taylor_cifar10(dataset_path,train=True,transform=train_transform,download=True,taylor_number=100)
        elif  training_config["dataset"] == "Cifar100":
            self.train_set = torchvision.datasets.CIFAR100(dataset_path,train=True,transform=train_transform,download=True)
            self.test_set = torchvision.datasets.CIFAR100(dataset_path,train=False,transform=test_transform,download=True)
            self.taylor_set = taylor_cifar10(dataset_path,train=True,transform=train_transform,download=True,taylor_number=100)
        elif  training_config["dataset"] == "Imagenet":
            self.train_set = torchvision.datasets.ImageFolder(os.path.join(dataset_path,"train"),train=True,transform=train_transform)
            self.test_set = torchvision.datasets.ImageFolder(os.path.join(dataset_path,"val"),train=False,transform=test_transform)
            self.taylor_set = taylor_imagenet(os.path.join(dataset_path,"train"),train=True,transform=train_transform)
        self.classes = len(self.train_set.classes)
        g = torch.Generator()
        g.manual_seed(seed)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers,worker_init_fn=seed_worker,generator=g)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers,worker_init_fn=seed_worker,generator=g)
        self.taylor_loader = torch.utils.data.DataLoader(self.taylor_set, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers,worker_init_fn=seed_worker,generator=g)

        print("==> Preparing K")
        if  training_config["model"] == "ResNet101":
            if pruning_config["Pruning"]["K_calculation"][1] == "Imagenet_K":
                pass
            else:
                pass
            
        elif training_config["model"] == "Mobilenetv2":
            if pruning_config["Pruning"]["K_calculation"][1] == "Imagenet_K":
                self.list_k = [1,16,48,2,34,57,4,36,79,42,2,270,5,2,80,431,185]
            else:
                raise NotImplementedError
        elif  training_config["model"] == "VGG16":
            if pruning_config["Pruning"]["K_calculation"][1] == "Imagenet_K":
                self.list_k = [10,1,16,15,17,69,31,70,28,52,13,241,241]
            elif pruning_config["Pruning"]["K_calculation"][1] == "Cifar100_K":
                self.list_k = [31,10,12,20,97,120,25,67,157,132,154,36,103]
            elif pruning_config["Pruning"]["K_calculation"][1] == "Cifar10_K":
                self.list_k = [14,5,12,42,15,9,83,139,92,134,223,209,82]

            
        # Netword preparation
        print("==> Preparing models")
        print(f"==> Using {self.device} mode")
        if  training_config["model"] == "ResNet101":
            self.net = ResNet101(num_classes=self.classes)
            self.teacher_net = ResNet101(num_classes=self.classes)
            
        elif  training_config["model"] == "Mobilenetv2":
            self.net = MobileNetV2(num_classes=self.classes)
            self.teacher_net = MobileNetV2(num_classes=self.classes)
            # net = MobileNetV2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        elif  training_config["model"] == "VGG16":
            self.net = VGG(num_class=self.classes)
            self.teacher_net = VGG(num_class=self.classes)
            
        self.net.to(self.device)
        self.net.load_state_dict(torch.load(pruning_config["Model"]["Pretrained_weight_path"])["state_dict"])
        self.teacher_net.to(self.device)

        self.teacher_net.load_state_dict(torch.load(pruning_config["Model"]["Pretrained_teacher_weight_path"])["state_dict"])
        self.teacher_net = frozen_layer(self.teacher_net)
        self.best_acc = 0
        self.pruning_method = pruning_config["Pruning"]["Pruning_method"]
        self.pruning_ratio_list = pruning_config["Pruning"]["Pruning_ratio"]
        self.pruning_ratio = self.pruning_ratio_list[0]
        if self.pruning_method == "L1norm":
            self.pruner = pruning_engine(self.pruning_method,self.pruning_ratio)
        
        self.distillation_temperature = training_config["distillation_temperature"]
        self.distillation_alpha = training_config["distillation_alpha"]

    def validation(self,criterion,save=True,optimizer=None):
        
        # loop over the dataset multiple times
        accuracy = 0
        running_loss = 0.0
        total = 0
        correct = 0
        self.net.eval()
        with tqdm(total=len(self.test_loader)) as pbar:
            with torch.no_grad():
                for i, data in enumerate(self.test_loader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)


                    # forward + backward + optimize
                    outputs = self.net(inputs)
                    _, predicted = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    accuracy = 100 * correct / total
                    pbar.update()
                    pbar.set_description_str("Acc: {:.3f} {}/{} | Loss: {:.3f}".format(accuracy,correct,total,loss/(i+1)))
                if save:
                    if not os.path.isdir(self.weight_path):
                        os.makedirs(self.weight_path)
                    check_point_path = os.path.join(self.weight_path,"Checkpoint.pt")
                    torch.save({"state_dict":self.net.state_dict(),"optimizer":optimizer.state_dict()},check_point_path)    
                if accuracy > self.best_acc:
                    if save:
                        self.best_acc = accuracy
                    
                        PATH = os.path.join(self.weight_path,f"Model@{self.model_name}_ACC@{self.best_acc}.pt")
                        torch.save(self.net.state_dict(), PATH)
                        print("Save: Acc "+str(self.best_acc))
                    else:
                        print("Finish Testing")
        return running_loss/len(self.test_loader),accuracy

    def train(self,epoch,optimizer,criterion,warmup_scheduler):
        # loop over the dataset multiple times
        running_loss = 0.0
        total = 0
        correct = 0
        self.net.train()
        self.teacher_net.eval()
        with tqdm(total=len(self.train_loader)) as pbar:
            for i, data in enumerate(self.train_loader, 0):
                
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)

                # Knowledge distillation
                with torch.no_grad():
                    teacher_outputs = self.teacher_net(inputs)


                loss = criterion(
                    F.log_softmax(outputs/self.distillation_temperature, dim=1),
                    F.softmax(teacher_outputs/self.distillation_temperature, dim=1)
                                )
                loss = loss * (self.distillation_alpha * self.distillation_temperature * self.distillation_temperature) 
                loss = loss + F.cross_entropy(outputs, labels) * (1. - self.distillation_alpha)
                loss.backward()
                optimizer.step()
                if epoch <= self.warmup_epoch:
                    warmup_scheduler.step()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                running_loss += loss.item()            
                
                
                accuracy = 100 * correct / total
                pbar.update()
                pbar.set_description_str("Epoch: {} | Acc: {:.3f} {}/{} | Loss: {:.3f}".format(epoch,accuracy,correct,total,running_loss/(i+1)))
    def OpCounter(self):
        input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
        macs, params = profile(self.net, inputs=(input, ))
        macs, params = clever_format([macs, params], "%.3f")
        return macs,params
    def pruning(self):
        raise Exception("Not Implemented")
    
    def retraining(self):
        raise Exception("Not Implemented")

    def hook_function(self,tool_net,forward_hook,backward_hook):
        raise Exception("Not Implemented")