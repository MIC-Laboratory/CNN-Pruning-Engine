import torch
import os
import sys

from .Kmean_base import Kmean_base
from ..L1norm.L1norm import L1norm
class K_L1norm(Kmean_base,L1norm):
    def __init__(self,list_k,pruning_ratio):
        """
        Initialize K_L1norm object.

        Args:
            None

        Return:
            None

        Logic:
            Initialize K_L1norm object.
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.list_k = list_k
        self.pruning_ratio = pruning_ratio
    def Kmean_L1norm(self,layer):
        """
        Apply K-L1norm pruning to the given layer.

        Args:
            layer: The layer to be pruned.

        Return:
            sorted_idx: The sorted indices of the most important filters.

        Logic:
            1. Clone the weight data of the layer to avoid modifying the original weights.
            2. Sort layer filter by the L1 norm and obtain the corresponding indices.
            3. Return the sorted indices, representing the most important filters based on K-L1norm.
        """
        weight = layer.weight.data.clone()
        sort_index = self.L1norm_pruning(layer)
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
