import torch
from copy import deepcopy
from .pruning_engine_base import pruning_engine_base
from .Pruning_criterion.L1norm.L1norm import L1norm
from .Pruning_criterion.Taylor.Taylor import Taylor
from .Pruning_criterion.KMean.K_L1norm import K_L1norm
from .Pruning_criterion.KMean.K_Taylor import K_Taylor
class pruning_engine(pruning_engine_base):
    def __init__(self,pruning_method,pruning_ratio = 0,individual = False,**kwargs):
        super().__init__(pruning_ratio,pruning_method)
        
        
        if (self.pruning_method == "L1norm"):
            l1norm_pruning = L1norm()
            self.pruning_criterion = l1norm_pruning.L1norm_pruning
        elif (self.pruning_method == "Taylor"):
            taylor_pruning = Taylor(
                tool_net=kwargs["tool_net"],
                total_layer=kwargs["total_layer"], 
                taylor_loader=kwargs["taylor_loader"],
                total_sample_size=kwargs["total_sample_size"], 
                hook_function=kwargs["hook_function"])
            taylor_pruning.clear_mean_gradient_feature_map()
            taylor_pruning.Taylor_add_gradient()
            taylor_pruning.store_grad_layer(kwargs["layer_store_private_variable"])
            self.pruning_criterion = taylor_pruning.Taylor_pruning
        elif (self.pruning_method == "K-L1norm"):
            self.K_L1norm_pruning = K_L1norm(list_k=kwargs["list_k"],pruning_ratio=self.pruning_ratio)
            self.K_L1norm_pruning.store_k_in_layer(kwargs["layer_store_private_variable"])
            
            self.pruning_criterion = self.K_L1norm_pruning.Kmean_L1norm
        elif (self.pruning_method == "K-Taylor"):
            taylor_pruning = Taylor(
                tool_net=kwargs["tool_net"],
                total_layer=kwargs["total_layer"], 
                taylor_loader=kwargs["taylor_loader"],
                total_sample_size=kwargs["total_sample_size"], 
                hook_function=kwargs["hook_function"])
            taylor_pruning.clear_mean_gradient_feature_map()
            taylor_pruning.Taylor_add_gradient()
            taylor_pruning.store_grad_layer(kwargs["layer_store_private_variable"])
            self.K_Taylor_pruning = K_Taylor(
                list_k=kwargs["list_k"],
                pruning_ratio=self.pruning_ratio,
                taylor_pruning=taylor_pruning.Taylor_pruning)
            self.K_Taylor_pruning.store_k_in_layer(kwargs["layer_store_private_variable"])
            self.pruning_criterion = self.K_Taylor_pruning.Kmean_Taylor

        self.sorted_idx_history = {
            "previous_layer":None,
            "current_layer":None
        }
        self.individual = individual


    def set_layer(self,layer,main_layer=False):
        
        try:
            self.copy_layer = deepcopy(layer)
            
            if main_layer:
                if self.individual:
                    self.sorted_idx_history = {
                        "previous_layer":None,
                        "current_layer":None
                    }
                self.sorted_idx_history["previous_layer"] = self.sorted_idx_history["current_layer"]
                self.sorted_idx_history["current_layer"] = None
                sorted_idx = self.pruning_criterion(self.copy_layer)
                number_pruning_filter = int(len(sorted_idx) * self.pruning_ratio)
                self.sorted_idx = sorted_idx[number_pruning_filter:]
                if (self.sorted_idx_history["previous_layer"] is None):
                    self.sorted_idx_history["previous_layer"] = self.sorted_idx 
                self.sorted_idx_history["current_layer"] = self.sorted_idx
            return True
        except Exception as e:
            print("Error:",e)
            return False
    
    def set_pruning_ratio(self,pruning_ratio):
        self.pruning_ratio = 1-pruning_ratio
        if "K_L1norm_pruning" in self.__dict__:
            self.K_L1norm_pruning.set_pruning_ratio(1-pruning_ratio)
        if "K_Taylor_pruning" in self.__dict__:
            self.K_Taylor_pruning.set_pruning_ratio(1-pruning_ratio)

    def get_sorted_idx(self):
        return self.sorted_idx_history

    def remove_conv_filter_kernel(self):
        if self.copy_layer.bias is not None:
            self.copy_layer.weight.data,self.copy_layer.bias.data = self.base_remove_filter_by_index(weight=self.copy_layer.weight.data.clone(),sorted_idx=self.sorted_idx_history["current_layer"],bias=self.copy_layer.bias.data.clone())
            self.copy_layer.weight.data = self.base_remove_kernel_by_index(weight=self.copy_layer.weight.data.clone(), sorted_idx=self.sorted_idx_history["previous_layer"])
            self.copy_layer.out_channels = self.copy_layer.weight.shape[0]
            self.copy_layer.in_channels = self.copy_layer.weight.shape[1]
        else:
            self.copy_layer.weight.data = self.base_remove_filter_by_index(weight=self.copy_layer.weight.data.clone(),sorted_idx=self.sorted_idx_history["current_layer"])
            self.copy_layer.weight.data = self.base_remove_kernel_by_index(weight=self.copy_layer.weight.data.clone(), sorted_idx=self.sorted_idx_history["previous_layer"])
            self.copy_layer.out_channels = self.copy_layer.weight.shape[0]
            self.copy_layer.in_channels = self.copy_layer.weight.shape[1]
        
        return self.copy_layer
    
    def remove_Bn(self,sorted_idx):        
        self.copy_layer.weight.data,\
        self.copy_layer.bias.data,\
        self.copy_layer.running_mean.data,\
        self.copy_layer.running_var.data = self.base_remove_filter_by_index(
            self.copy_layer.weight.data.clone(), 
            sorted_idx,
            bias=self.copy_layer.bias.data.clone(),
            mean=self.copy_layer.running_mean.data.clone(),
            var=self.copy_layer.running_var.data.clone()
            )
        self.copy_layer.num_features = self.copy_layer.weight.shape[0]
        return self.copy_layer
    
    def remove_filter_by_index(self,sorted_idx,linear=False,group=False):
        if self.copy_layer.bias is not None:

            self.copy_layer.weight.data,\
            self.copy_layer.bias.data = self.base_remove_filter_by_index(
                weight=self.copy_layer.weight.data.clone(),
                sorted_idx=sorted_idx,
                bias=self.copy_layer.bias.data.clone(),
                linear=linear
                )
        else:
            self.copy_layer.weight.data = self.base_remove_filter_by_index(
                weight=self.copy_layer.weight.data.clone(),
                sorted_idx=sorted_idx,
                linear=linear
                )
        if linear:
            self.copy_layer.out_features = self.copy_layer.weight.shape[0]
        else:
            self.copy_layer.out_channels = self.copy_layer.weight.shape[0]
        
        if group:
            self.copy_layer.groups = self.copy_layer.weight.shape[0]
            self.copy_layer.in_channels = self.copy_layer.weight.shape[1]
            self.copy_layer.out_channels = self.copy_layer.weight.shape[0]
        return self.copy_layer

    def remove_kernel_by_index(self,sorted_idx,linear=False):
        self.copy_layer.weight.data = self.base_remove_kernel_by_index(
            weight=self.copy_layer.weight.data.clone(),
            sorted_idx=sorted_idx,
            linear=linear
            
            )
        if linear:
            self.copy_layer.in_features = self.copy_layer.weight.shape[1]
        else:
            self.copy_layer.in_channels = self.copy_layer.weight.shape[1]

        return self.copy_layer