from ..KMean.Kmean_base import Kmean_base
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import torch
import numpy as np
class K_Distance(Kmean_base):
    def __init__(self,list_k,pruning_ratio):
        """
        Initialize K_Distance object.

        Args:
            None

        Return:
            None

        Logic:
            Initialize K_Distance object.
        """
        Kmean_base.__init__(K_Distance,list_k,pruning_ratio)
    def Kmean_Distance(self,layer):
        n_clusters = layer.k_value
        num_filter = layer.weight.data.size()[0]
        m_weight_vector = layer.weight.reshape(num_filter, -1).cpu().detach().numpy()
        bn = layer.bn.cpu().detach().numpy().reshape(-1,1)
        m_weight_vector = m_weight_vector+bn
        pca = PCA(n_components=0.8).fit(m_weight_vector)
        m_weight_vector = pca.fit_transform(m_weight_vector)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0,n_init='auto').fit(m_weight_vector)
        distance_set = kmeans.fit_transform(m_weight_vector)

        num_filter_list = [i for i in range(num_filter)]
        distance = distance_set[num_filter_list,kmeans.labels_]
        unique, index,counts = np.unique(kmeans.labels_, return_counts=True,return_index=True)
        lock_group = index[counts==1]
        distance[lock_group] = 1e10
        distance = torch.from_numpy(distance)
        sorted_importance, sorted_idx = torch.sort(distance, dim=0, descending=True)
        return sorted_idx
