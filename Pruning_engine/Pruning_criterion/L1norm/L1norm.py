import torch

class L1norm:
    def L1norm_pruning(self,layer):
        """
        Apply L1 norm pruning to the given layer.

        Args:
            layer: The layer to be pruned.

        Return:
            sorted_indices: The sorted indices of important filters.

        Logic:
        1. Clone the weight data of the layer to avoid modifying the original weights.
        2. Check the shape of the weight tensor to determine if it is a convolutional layer.
        3. Calculate the importance of each filter by summing the absolute values of the weights along the appropriate dimensions.
           - For a 4-dimensional weight tensor, the sum is calculated along dimensions (1, 2, 3).
           - For a 2-dimensional weight tensor, the sum is calculated along dimension 0 (channels).
        4. Sort the importance values and obtain the corresponding indices in descending order.
        5. Return the sorted indices

    Note:
        The importance values indicate the amount of contribution of each filter to the overall model's performance.
        Filters with higher importance values are considered more significant and likely to be kept during pruning.
        """
        weight = layer.weight.data.clone()
        if len(weight.shape) == 4:
            importance = torch.sum(torch.abs(weight),dim=(1,2,3))
        else:
            
            importance = torch.sum(torch.abs(weight),dim=0)
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        return sorted_idx