python3 Mobilenetv2_pruning.py \
--weight_path=Weights/Cifar10/mobilenetv2/acc92.31%_mobilenetv2 \
--dataset=Cifar10 \
--dataset_path=/home/zhenyulin/Training_data \
--pruning_mode=Fullayer \
--pruning_method=K-L1norm