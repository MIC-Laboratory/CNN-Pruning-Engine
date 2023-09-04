from sklearn.cluster import KMeans
import torch
import copy
from sklearn.decomposition import PCA
class Kmean_base:
    def __init__(self):
        pass
    def Kmean(self,weight,sort_index,k,output_channel):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        num_filter = weight.data.size()[0]
        remove_filter = num_filter - output_channel
        
        n_clusters = k
        m_weight_vector = weight.reshape(num_filter, -1).cpu()

        pca = PCA(n_components=0.7).fit(m_weight_vector)
        m_weight_vector = pca.fit_transform(m_weight_vector)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(m_weight_vector)
        print("K:",n_clusters)
        labels = kmeans.labels_
        group = [[] for _ in range(n_clusters)]
        for idx in range(num_filter):
            group[labels[idx]].append(idx)
        lock_group_index = []
        copy_group = copy.deepcopy(group)
        for filter_index_group in copy_group:
            if len(filter_index_group) == 1:
                group.remove(filter_index_group)

        # The reminding item in group can be pruned by some crition
        pruning_index_group = []
        pruning_left_index_group = [[] for _ in range(len(group))]
        total_left_filter = sum(len(filter_index_group)
                                for filter_index_group in group)
        percentage_group = [int(
            100*(len(filter_index_group)/total_left_filter)) for filter_index_group in group]
        pruning_amount_group = [
            int(remove_filter*(percentage/100)) for percentage in percentage_group]
        sorted_idx_origin = sort_index
        for counter, filter_index_group in enumerate(group, 0):
            temp = copy.deepcopy(filter_index_group)
            temp.sort(key=lambda e: (list(sorted_idx_origin).index(e),e) if e in list(sorted_idx_origin)  else (len(list(sorted_idx_origin)),e))
            sorted_idx = torch.tensor(temp,device=device)
            filetr_index_group_temp = copy.deepcopy(list(sorted_idx))
            
            for sub_index in sorted_idx[len(sorted_idx)-pruning_amount_group[counter]:]:
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

        return torch.tensor(pruning_index_group).to(device)