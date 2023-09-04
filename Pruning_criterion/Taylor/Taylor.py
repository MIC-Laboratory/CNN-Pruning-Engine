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
                    self.mean_feature_map[feature_map_layer] = output.detach()/(self.total_sample_size)
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
                        self.mean_gradient[gradient_layer] = grad.detach()/(self.total_sample_size)
                else:
                    if len(self.taylor_loader) > 1:
                        self.mean_gradient[gradient_layer] = torch.add(self.mean_gradient[gradient_layer],torch.sum(grad.detach(),dim=(0))/(self.total_sample_size))
                    else:
                        self.mean_gradient[gradient_layer] = torch.add(self.mean_gradient[gradient_layer],grad.detach()/(self.total_sample_size))
                gradient_layer-=1
            output.register_hook(_store_grad)
        self.tool_net = self.hook_function(self.tool_net,forward_hook,backward_hook)
        
        # for block in range(self.tool_net):
        #     layer = tool_net.blocks[block+1].mobile_inverted_conv
        #     if block_channel_origin[block] == "_":
        #         continue
        #     if type(layer).__name__ == "ZeroLayer":
        #         print("Skip Pruning: Zero Layer")
        #         continue
        #     layer.depth_conv.conv.register_forward_hook(forward_hook)
        #     layer.depth_conv.conv.register_forward_hook(backward_hook)
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
        
        cam_grad = layer.gradient*layer.feature_map
        # cam_grad = self.mean_gradient[index]*self.mean_feature_map[index]
        cam_grad = torch.abs(cam_grad)
        criteria_for_layer = cam_grad / (torch.linalg.norm(cam_grad) + 1e-8)
        
        importance = torch.sum(criteria_for_layer,dim=(1,2))
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        return sorted_idx