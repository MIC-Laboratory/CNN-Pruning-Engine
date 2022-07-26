python3 Resnet18_pruning.py \
--weight_path=Weights/Cifar10/ \
--dataset=Imagenet \
--dataset_path=/home/zhenyulin/Training_data/imagenet-train \
--pruning_mode=Fullayer \
--pruning_method=K-L1norm