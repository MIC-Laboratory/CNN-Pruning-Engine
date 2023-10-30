from ..KMean.Kmean_base import Kmean_base
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import distance
import torch
import numpy as np
import copy
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
    # data point distance to the cluster center
    
    # def Kmean_Distance(self,layer):
    #     n_clusters = layer.k_value
    #     num_filter = layer.weight.data.size()[0]
    #     m_weight_vector = layer.weight.reshape(num_filter, -1).cpu().detach().numpy()
    #     bn = layer.bn.cpu().detach().numpy().reshape(-1,1)
    #     m_weight_vector = m_weight_vector+bn
    #     pca = PCA(n_components=0.8).fit(m_weight_vector)
    #     m_weight_vector = pca.fit_transform(m_weight_vector)
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=0,n_init='auto').fit(m_weight_vector)
    #     distance_set = kmeans.fit_transform(m_weight_vector)

    #     num_filter_list = [i for i in range(num_filter)]
    #     distance = distance_set[num_filter_list,kmeans.labels_]
    #     unique, index,counts = np.unique(kmeans.labels_, return_counts=True,return_index=True)
    #     lock_group = index[counts==1]
    #     distance[lock_group] = 1e10
    #     distance = torch.from_numpy(distance)
    #     sorted_importance, sorted_idx = torch.sort(distance, dim=0, descending=True)
    #     return sorted_idx

    # pairwise distance
    def Kmean_Distance(self,layer):
        n_clusters = layer.k_value
        num_filter = layer.weight.data.size()[0]
        m_weight_vector = layer.weight.reshape(num_filter, -1).cpu().detach().numpy()
        bn = layer.bn.cpu().detach().numpy().reshape(-1,1)
        m_weight_vector = m_weight_vector+bn
        pca = PCA(n_components=0.8).fit(m_weight_vector)
        m_weight_vector = pca.fit_transform(m_weight_vector)
        output_channel = int(num_filter * self.pruning_ratio)
        remove_filter = num_filter - output_channel
        kmeans = KMeans(n_clusters=n_clusters, random_state=0,n_init='auto').fit(m_weight_vector)
        group = [[] for _ in range(n_clusters)]
        origin_idx = [i for i in range(num_filter)]
        origin_idx = torch.tensor(origin_idx)
        for i in range(n_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            cluster_data = m_weight_vector[cluster_indices]
            
            # Check if the cluster is not empty
            if cluster_data.size > 0:
                pairwise_distances = distance.pdist(cluster_data)
                pairwise_distances_matrix = distance.squareform(pairwise_distances)
                
                # Calculate average pairwise distance for each sample
                avg_pairwise_distances = np.mean(pairwise_distances_matrix, axis=1)
                
                # Sort samples based on average pairwise distances
                sorted_indices = np.argsort(avg_pairwise_distances)
                sorted_cluster_indices = cluster_indices[sorted_indices]
                sorted_cluster_indices = sorted_cluster_indices[::-1]
                sorted_cluster_indices = list(sorted_cluster_indices)
                for index in sorted_cluster_indices:
                    group[i].append(index)


        # The reminding item in group can be pruned by some crition
        pruning_index_group = []
        pruning_left_index_group = [[] for _ in range(len(group))]
        total_left_filter = sum(len(filter_index_group)
                                for filter_index_group in group)
        percentage_group = [int(
            100*(len(filter_index_group)/total_left_filter)) for filter_index_group in group]
        pruning_amount_group = [
            int(remove_filter*(percentage/100)) for percentage in percentage_group]
        for counter, filter_index_group in enumerate(group, 0):
            temp = copy.deepcopy(filter_index_group)
            temp = torch.tensor(temp,device=self.device)
            filetr_index_group_temp = copy.deepcopy(list(temp))
            
            for sub_index in temp[len(temp)-pruning_amount_group[counter]:]:
                if len(filetr_index_group_temp) == 1:
                    continue
                pruning_index_group.append(filetr_index_group_temp.pop(filetr_index_group_temp.index(sub_index)))
            for left_index in filetr_index_group_temp:
                pruning_left_index_group[counter].append(left_index)
        # first one is the least important weight and the last one is the most important weight
        while (len(pruning_index_group) < remove_filter):
            pruning_amount = len(pruning_index_group)
            for left_index in pruning_left_index_group:
                if (len(left_index) <= 1):
                    continue
                if (len(pruning_index_group) >= remove_filter):
                    break
                pruning_index_group.append(left_index.pop(-1))
            if (pruning_amount >= len(pruning_index_group)):
                raise ValueError('infinity loop')

        pruning_index = torch.tensor(pruning_index_group).to(self.device)
        keep_index = [i.item() for i in origin_idx if i not in pruning_index]
        keep_index = torch.as_tensor(keep_index,device=self.device)

        return torch.cat((keep_index,pruning_index))
