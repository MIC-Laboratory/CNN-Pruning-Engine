import torch

# Set the unimportant filter and kernel to 0
# class pruning_engine_base:
#     def __init__(self,pruning_ratio,pruning_method):
        

#         self.pruning_ratio = 1-pruning_ratio
        
#         self.pruning_method = pruning_method
#         self.mask_number = 0
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#     def base_remove_filter_by_index(self,weight,remove_filter_idx,bias=None,mean=None,var=None,linear=False):       
#         if mean is not None:
#             mask_tensor = torch.tensor(self.mask_number,device=self.device)
#             for idx in remove_filter_idx:
#                 weight[idx.item()] = mask_tensor
#                 bias[idx.item()] = mask_tensor
#                 mean[idx.item()] = mask_tensor 
#                 var[idx.item()] = mask_tensor
#             return weight,bias,mean,var
#         elif bias is not None:
#             mask_tensor = torch.tensor(self.mask_number,device=self.device)
#             mask_tensor = mask_tensor.repeat(list(weight[0].size()))
#             bias_mask_tensor = torch.tensor(self.mask_number,device=self.device)
#             for idx in remove_filter_idx:
#                 weight[idx.item()] = mask_tensor
#                 bias[idx.item()] = bias_mask_tensor
            
#             return weight,bias
#         else:
#             mask_tensor = torch.tensor(self.mask_number,device=self.device)
#             mask_tensor = mask_tensor.repeat(list(weight[0].size()))
#             for idx in remove_filter_idx:
#                 weight[idx.item()] = mask_tensor
            
#             return weight

#     def base_remove_kernel_by_index(self,weight,remove_filter_idx,linear=False):
#         mask_tensor = torch.tensor(self.mask_number,device=self.device)
#         mask_tensor = mask_tensor.repeat(list(weight[0][0].size()))
#         for idx in remove_filter_idx:
#             weight[:,idx.item()] = mask_tensor
        
#         return weight








# Real Remove filter and kernel
class pruning_engine_base:
    def __init__(self,pruning_ratio,pruning_method):
        
        """
        Initialize the pruning engine base class.

        Args:
        - pruning_ratio: The pruning ratio to be applied.
        - pruning_method: The chosen pruning method.

        Return: None
        """
        self.pruning_ratio = 1-pruning_ratio
        
        self.pruning_method = pruning_method
        self.mask_number = 1e10
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def base_remove_filter_by_index(self,weight,remove_filter_idx,bias=None,mean=None,var=None,linear=False):       
        """
        Remove the specified filters from the layer's weight, bias, mean, and var tensors.

        Args:
        - weight: The weight tensor of the layer.
        - remove_filter_idx: The indices of filters to be removed.
        - bias: The bias tensor of the layer.
        - mean: The mean tensor of the Batch Normalization layer.
        - var: The variance tensor of the Batch Normalization layer.
        - linear: A flag indicating whether the layer is a Linear layer.

        Return:
        - weight: The updated weight tensor after removing the filters.
        - bias: The updated bias tensor after removing the filters.
        - mean: The updated mean tensor after removing the filters.
        - var: The updated variance tensor after removing the filters.
        """
        if mean is not None:
            mask_tensor = torch.tensor(self.mask_number,device=self.device)
            for idx in remove_filter_idx:
                weight[idx.item()] = mask_tensor
                bias[idx.item()] = mask_tensor
                mean[idx.item()] = mask_tensor 
                var[idx.item()] = mask_tensor
            weight = weight[weight != mask_tensor]
            bias = bias[bias != mask_tensor]
            mean = mean[mean != mask_tensor]
            var = var[var != mask_tensor]
            return weight,bias,mean,var
        elif bias is not None:
            mask_tensor = torch.tensor(self.mask_number,device=self.device)
            mask_tensor = mask_tensor.repeat(list(weight[0].size()))
            bias_mask_tensor = torch.tensor(self.mask_number,device=self.device)
            for idx in remove_filter_idx:
                weight[idx.item()] = mask_tensor
                bias[idx.item()] = bias_mask_tensor
            if linear is False:
                nonMaskRows_weight = abs(torch.abs(weight).sum(dim=(1,2,3)) - torch.abs(mask_tensor).sum(dim=(0,1,2))) > self.mask_number
            else:
                nonMaskRows_weight = abs(torch.abs(weight).sum(dim=1) - torch.abs(mask_tensor).sum(dim=0)) > self.mask_number
            weight = weight[nonMaskRows_weight]
            bias = bias[bias != self.mask_number]
            return weight,bias
        else:
            mask_tensor = torch.tensor(self.mask_number,device=self.device)
            mask_tensor = mask_tensor.repeat(list(weight[0].size()))
            for idx in remove_filter_idx:
                weight[idx.item()] = mask_tensor
            if linear is False:
                nonMaskRows_weight = abs(torch.abs(weight).sum(dim=(1,2,3)) - torch.abs(mask_tensor).sum(dim=(0,1,2))) > self.mask_number
            else:
                nonMaskRows_weight = abs(torch.abs(weight).sum(dim=1) - torch.abs(mask_tensor).sum(dim=1)) > self.mask_number
            weight = weight[nonMaskRows_weight]
            return weight

    def base_remove_kernel_by_index(self,weight,remove_filter_idx,linear=False):
        """
        Remove the specified kernels from the layer's weight tensor.

        Args:
        - weight: The weight tensor of the layer.
        - remove_filter_idx: The indices of kernels to be removed.
        - linear: A flag indicating whether the layer is a Linear layer.

        Return:
        - weight: The updated weight tensor after removing the kernels.
        """
        mask_tensor = torch.tensor(self.mask_number,device=self.device)
        mask_tensor = mask_tensor.repeat(list(weight[0][0].size()))
        for idx in remove_filter_idx:
            weight[:,idx.item()] = mask_tensor
        if (len(remove_filter_idx) != 0 and linear == False):
            nonMaskRows_weight = abs(torch.abs(weight).sum(dim=(2,3)) - torch.abs(mask_tensor).sum(dim=(0,1))) > 0.0001 
            weight = weight[:,nonMaskRows_weight[0]]
        if (linear != False):
            weight = weight[:,weight[1]!=mask_tensor]
        return weight



