import torch
import os
import sys

from .Kmean_base import Kmean_base
from ..Taylor.Taylor import Taylor

class K_Taylor(Kmean_base,Taylor):
    def __init__(self,list_k,pruning_ratio,tool_net,taylor_loader,total_layer,total_sample_size,hook_function,layer_store_grad_featuremap):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.list_k = list_k
        self.pruning_ratio = pruning_ratio
        Taylor.__init__(K_Taylor,tool_net,taylor_loader,total_layer,total_sample_size,hook_function)
        self.clear_mean_gradient_feature_map()
        self.Taylor_add_gradient()
        self.store_grad_layer(layer_store_grad_featuremap)
    def Kmean_Taylor(self,layer):
        weight = layer.weight.data.clone()
        sort_index = self.Taylor_pruning(layer)
        k = layer.k_value

        output_channel = int(weight.shape[0] * self.pruning_ratio)
        
        
        pruning_index =  self.Kmean(weight,sort_index,k,output_channel)
        """
        using l1norm to sort the pruning index, and put them to the end of sorted_idx
        indicate they are not important
        However, experiment find out it doesn't help
        so I comment out them
        """
        # pruning_weight = weight[pruning_index,:,:,:]
        # important = torch.sum(torch.abs(pruning_weight),dim=(1,2,3))
        # pruning_weight,pruning_index = torch.sort(important)
        keep_index = [i.item() for i in sort_index if i not in pruning_index]
        keep_index = torch.as_tensor(keep_index,device=self.device)

        return torch.cat((keep_index,pruning_index))

    def set_pruning_ratio(self,pruning_ratio):
        self.pruning_ratio = pruning_ratio

    def store_k_in_layer(self,layers):
        for layer in range(len(layers)):
            layers[layer].__dict__["k_value"] = self.list_k[layer]