import torch
import os
import sys

from .Kmean_base import Kmean_base
from ..Taylor.Taylor import Taylor

class K_Taylor(Kmean_base,Taylor):
    def __init__(self,list_k,pruning_ratio,tool_net,taylor_loader,total_layer,total_sample_size,hook_function,layer_store_grad_featuremap):

        Taylor.__init__(K_Taylor,tool_net,taylor_loader,total_layer,total_sample_size,hook_function)
        Kmean_base.__init__(K_Taylor,list_k,pruning_ratio)
        self.clear_mean_gradient_feature_map()
        self.Taylor_add_gradient()
        self.store_grad_layer(layer_store_grad_featuremap)
    def Kmean_Taylor(self,layer):
        sort_index = self.Taylor_pruning(layer)
        k = layer.k_value
        weight = layer.weight.data.clone()
        output_channel = int(weight.shape[0] * self.pruning_ratio)
        weight = weight.reshape(weight.shape[0],-1).cpu().detach().numpy()
        
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


