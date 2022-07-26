python3 Vgg_pruning.py \
--weight_path=Weights/Cifar10/vgg16/acc94.16%_vgg16 \
--dataset=Cifar10 \
--dataset_path=/home/zhenyulin/Training_data/ \
--pruning_mode=Fullayer \
--pruning_method=K-L1norm