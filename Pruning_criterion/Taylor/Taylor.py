import torch
import time
import gc
from copy import deepcopy
from tqdm import tqdm

class Taylor:
    def __init__(self,tool_net,taylor_loader,total_layer,total_sample_size,hook_function):
        self.mean_feature_map = ["" for i in range(total_layer)]
        self.mean_gradient = ["" for i in range(total_layer)]
        self.taylor_loader = taylor_loader
        self.total_layer = total_layer
        self.total_sample_size = total_sample_size
        self.tool_net = tool_net
        self.hook_function = hook_function
        
    def Taylor_add_gradient(self):
        feature_map_layer = 0
        taylor_loader_iter = iter(self.taylor_loader)
        gradient_layer = self.total_layer-1
        def forward_hook(model, input, output):
            nonlocal feature_map_layer
            if (feature_map_layer >= self.total_layer):
                feature_map_layer = 0
            if self.mean_feature_map[feature_map_layer] == "":
                if len(self.taylor_loader) > 1:
                    self.mean_feature_map[feature_map_layer] = torch.sum(output.detach(),dim=(0))/(self.total_sample_size)
                else:
                    self.mean_feature_map[feature_map_layer] = torch.mean(output.detach(),dim=(0))
            else:
                if len(self.taylor_loader) > 1:
                    self.mean_feature_map[feature_map_layer] = torch.add(self.mean_feature_map[feature_map_layer],torch.sum(output.detach(),dim=(0))/(self.total_sample_size))
                else:
                    self.mean_feature_map[feature_map_layer] = torch.add(self.mean_feature_map[feature_map_layer],output.detach()/(self.total_sample_size))
            feature_map_layer+=1
        def backward_hook(model,input,output):
            if not hasattr(output, "requires_grad") or not output.requires_grad:
                # You can only register hooks on tensor requires grad.
                return
            def _store_grad(grad):
                nonlocal gradient_layer
                if (gradient_layer < 0):
                    gradient_layer = self.total_layer-1
                if self.mean_gradient[gradient_layer] == '':
                    if len(self.taylor_loader) > 1:
                        self.mean_gradient[gradient_layer] = torch.sum(grad.detach(),dim=(0))/(self.total_sample_size)
                    else:
                        self.mean_gradient[gradient_layer] = torch.mean(grad.detach(),dim=(0))
                else:
                    if len(self.taylor_loader) > 1:
                        self.mean_gradient[gradient_layer] = torch.add(self.mean_gradient[gradient_layer],torch.sum(grad.detach(),dim=(0))/(self.total_sample_size))
                    else:
                        self.mean_gradient[gradient_layer] = torch.add(self.mean_gradient[gradient_layer],grad.detach()/(self.total_sample_size))
                gradient_layer-=1
            output.register_hook(_store_grad)
        self.tool_net = self.hook_function(self.tool_net,forward_hook,backward_hook)
        

        optimizer = torch.optim.Adam(self.tool_net.parameters())
        loss = torch.nn.CrossEntropyLoss()
        with tqdm(total=len(self.taylor_loader)) as pbar:
            
            for _ in range(len(self.taylor_loader)):
                start = time.time()
                self.train(self.tool_net,optimizer,taylor_loader_iter,loss)
                gc.collect()
                pbar.update()
                pbar.set_description_str(f"training time: {time.time()-start}")
    
    def store_grad_layer(self,layers):
        # copy_layers = deepcopy(layers)
        for layer in range(len(layers)):
            layers[layer].__dict__["feature_map"] = self.mean_feature_map[layer]
            layers[layer].__dict__["gradient"] = self.mean_gradient[layer]


        # return copy_layers
    def clear_mean_gradient_feature_map(self):
        self.mean_gradient = ["" for _ in range(self.total_layer)]
        self.mean_feature_map = ["" for _ in range(self.total_layer)]


    def train(self,network,optimizer,dataloader_iter,criterion):
    
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        network.train()
        network.to(device)
        inputs, labels = next(dataloader_iter)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

    def Taylor_pruning(self,layer):
        
        criteria_for_layer = torch.mul(layer.gradient,layer.feature_map)
        criteria_for_layer = criteria_for_layer.view([*criteria_for_layer.shape[:2], -1])
        criteria_for_layer = criteria_for_layer.mean(dim=2)
        criteria_for_layer = torch.abs(criteria_for_layer)
        importance = criteria_for_layer.mean(dim=1) 
        """
        That's the old algorithm, it seems not working, comment out
        """
        # criteria_for_layer = torch.abs(criteria_for_layer)
        # criteria_for_layer = criteria_for_layer / (torch.linalg.norm(criteria_for_layer) + 1e-8)
        
        # importance = torch.mean(criteria_for_layer,dim=(1,2))
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        return sorted_idx