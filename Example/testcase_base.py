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
from torchvision.models import vgg16_bn,VGG16_BN_Weights
from torchvision.models import resnet101, ResNet101_Weights

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
        # self.device = torch.device('cpu')
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
        if training_config["dataset"] == "Imagenet":
            crop = [
                transforms.Resize(256),
                transforms.RandomCrop(self.input_size),
            ]
        else:
            crop = transforms.RandomCrop(self.input_size,padding=4),
            
        train_transform = transforms.Compose(
            [
            *crop,
            transforms.RandomHorizontalFlip(),
            # transforms.autoaugment.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            transforms.Normalize(mean=dataset_mean,std=dataset_std)
            ])

        test_transform = transforms.Compose([
            *crop,
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
            self.taylor_set = taylor_cifar100(dataset_path,train=True,transform=train_transform,download=True,taylor_number=100)
        elif  training_config["dataset"] == "Imagenet":
            self.train_set = torchvision.datasets.ImageFolder(os.path.join(dataset_path,"train"),transform=train_transform)
            self.test_set = torchvision.datasets.ImageFolder(os.path.join(dataset_path,"val"),transform=test_transform)
            self.taylor_set = taylor_imagenet(os.path.join(dataset_path,"train"),transform=train_transform,data_limit=100)
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
                # We haven't calculate the k for resnet, so use silhouette_score to calculate the best k
                # to use silhouette_score, we set the list_k to -1 for all layer
                self.list_k = [-1 for _ in range(33)]
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
            if training_config["dataset"] == "Imagenet":
                self.net = resnet101(ResNet101_Weights)
                self.teacher_net = resnet101(ResNet101_Weights)
            else:
                self.net = ResNet101(num_classes=self.classes)
                self.teacher_net = ResNet101(num_classes=self.classes)
            
        elif  training_config["model"] == "Mobilenetv2":
            self.net = MobileNetV2(num_classes=self.classes)
            self.teacher_net = MobileNetV2(num_classes=self.classes)
            # net = MobileNetV2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        elif  training_config["model"] == "VGG16":
            if training_config["dataset"] == "Imagenet":
                self.net = vgg16_bn(VGG16_BN_Weights)
                self.teacher_net = vgg16_bn(VGG16_BN_Weights)
            else:
                self.net = VGG(num_class=self.classes)
                self.teacher_net = VGG(num_class=self.classes)
        if training_config["dataset"] != "Imagenet": 
            if "state_dict" in torch.load(pruning_config["Model"]["Pretrained_weight_path"]).keys():
                self.net.load_state_dict(torch.load(pruning_config["Model"]["Pretrained_weight_path"])["state_dict"])
                self.teacher_net.load_state_dict(torch.load(pruning_config["Model"]["Pretrained_teacher_weight_path"])["state_dict"]) 
            else:
                self.net.load_state_dict(torch.load(pruning_config["Model"]["Pretrained_weight_path"]))
                self.teacher_net.load_state_dict(torch.load(pruning_config["Model"]["Pretrained_teacher_weight_path"])) 
        self.net.to(self.device)
        
        self.teacher_net.to(self.device)

        
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
        num_macs, num_params = profile(self.net, inputs=(input, ))
        macs, params = clever_format([num_macs, num_params], "%.3f")
        return macs,params,num_macs,num_params
    
    def retraining(self):
        training_criterion = nn.KLDivLoss()
        testing_criterion = nn.CrossEntropyLoss()

        print("==> Base validation acc:")
        loss,accuracy = self.validation(criterion=testing_criterion,save=False)

        print("==> Start pruning")
        optimizer = optim.SGD(self.net.parameters(), lr=self.lr_rate,momentum=self.momentum,weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.training_epoch)
        warmup_scheduler = WarmUpLR(optimizer, len(self.train_loader) * self.warmup_epoch)

        print("==> Start retraining")
        for epoch in range(self.training_epoch + self.warmup_epoch):
            self.train(
                epoch, 
                optimizer=optimizer,
                criterion=training_criterion,
                warmup_scheduler=warmup_scheduler
                )
            loss,accuracy = self.validation(
                criterion=testing_criterion,
                optimizer=optimizer,
                )
            if (epoch > self.warmup_epoch):
                scheduler.step()
            self.writer.add_scalar('Test/Loss', loss, epoch)
            self.writer.add_scalar('Test/ACC', accuracy, epoch)
        self.writer.close()
        print("==> Finish")
    # =========================================> Extra Experiment function

    def config_pruning(self):


        print("Before pruning:",self.OpCounter())
        testing_criterion = nn.CrossEntropyLoss()

        print("==> Base validation acc:")
        loss,accuracy = self.validation(criterion=testing_criterion,save=False)
        # vgg_testcase.pruning()
        # vgg_testcase.retraining()
        loss,accuracy = self.validation(criterion=testing_criterion,save=False)
        print("After pruning:",self.OpCounter())

    def layerwise_pruning(self):
        pruning_ratio_list_reference = deepcopy(self.pruning_ratio_list)
        vgg_testcase_net_reference = deepcopy(self.net)
        
        for layer_idx in range(len(pruning_ratio_list_reference)):
            self.pruning_ratio_list = deepcopy(pruning_ratio_list_reference)
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_path,"Layer"+str(layer_idx)))
            accuracy_list = []
            mac_list = []
            for pruning_percentage in range(10):
                vgg_testcase.net = deepcopy(vgg_testcase_net_reference)
                vgg_testcase.pruning_ratio_list[layer_idx] = round(pruning_percentage/10,1)

                print("Pruning Ratio:",self.pruning_ratio_list[layer_idx])

                print("Before pruning:",self.OpCounter())
                testing_criterion = nn.CrossEntropyLoss()

                print("==> Base validation acc:")
                loss,accuracy = self.validation(criterion=testing_criterion,save=False)
                self.pruning()
                loss,accuracy = self.validation(criterion=testing_criterion,save=False)
                print("After pruning:",self.OpCounter())
                _,_,mac,param = self.OpCounter()
                self.writer.add_scalar('Test/ACC', accuracy, pruning_percentage)
                self.writer.add_scalar('Test/Mac', mac, pruning_percentage)
                self.writer.add_scalar('Test/Param', param, pruning_percentage)
                self.writer.add_scalar('Test/ACC', accuracy, pruning_percentage)
                
                accuracy_list.append(accuracy)
                mac_list.append(float(vgg_testcase.OpCounter()[0].rstrip('M')))
            
            # with open(vgg_testcase.log_path+"pruning_ratio.txt","a") as f:
            #     f.write(f"\n===============>Layer{layer_idx} Pruning Ratio==================>"+str(calculate_pruning_ratio(accuracy_list,mac_list,13)))
            self.writer.close()

    def fullayer_pruning(self):

        vgg_testcase_net_reference = deepcopy(self.net)

        for pruning_percentage in range(20):
            for layer_idx in range(len(deepcopy(self.pruning_ratio_list))):
                self.pruning_ratio_list[layer_idx] = round(pruning_percentage/20,1)
            self.net = deepcopy(vgg_testcase_net_reference)
            print("Before pruning:",self.OpCounter())
            testing_criterion = nn.CrossEntropyLoss()

            print("==> Base validation acc:")
            loss,accuracy = self.validation(criterion=testing_criterion,save=False)
            self.pruning()
            loss,accuracy = self.validation(criterion=testing_criterion,save=False)
            print("After pruning:",self.OpCounter())
            _,_,mac,param = self.OpCounter()
            self.writer.add_scalar('Test/ACC', accuracy, pruning_percentage)
            self.writer.add_scalar('Test/Mac', mac, pruning_percentage)
            self.writer.add_scalar('Test/Param', param, pruning_percentage)
        self.writer.close()

    # =============== Formula: 
    # For each layer, the average Mac decrease,
    # divided by the average accuracy loss.
    # to norm between 0 and 1, you will need first run
    # with return raw_pruning_score to get the min and max value for raw_pruining score
    # Then fit in to this formula: (value - min) / (max - min)
    def calculate_pruning_ratio(self,accuracy_list,mac_list,num_conv_layer):

        accuracy_loss_list = []
        mac_loss_list = []

        for list_idx in range(len(accuracy_list)-1,0,-1):
            accuracy_loss_list.append(abs(accuracy_list[list_idx] - accuracy_list[list_idx-1]))
            mac_loss_list.append(abs(mac_list[list_idx] - mac_list[list_idx-1]))
        
        average_accuracy_loss = sum(accuracy_loss_list)/len(accuracy_loss_list)
        average_mac_loss = sum(mac_loss_list)/len(mac_loss_list)

        raw_pruning_score = round(average_mac_loss/average_accuracy_loss,2)
        # raw_pruning_score = round(average_accuracy_loss/average_mac_loss,2)
        return raw_pruning_score





    def pruning(self):
        raise Exception("Not Implemented")
    
    def hook_function(self,tool_net,forward_hook,backward_hook):
        raise Exception("Not Implemented")