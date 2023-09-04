import torch
import numpy
import random
from copy import deepcopy

def frozen_layer(layer):
    copy_layer = deepcopy(layer)
    for param in copy_layer.parameters():
        param.requires_grad=False
    return copy_layer

def deFrozen_layer(layer):
    copy_layer = deepcopy(layer)
    for param in copy_layer.parameters():
        param.requires_grad=True
    return copy_layer

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

