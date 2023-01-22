python3 Resnet18_pruning.py \
--weight_path=Weights/Cifar100/resnet18/acc68.35%_resnet18 \
--dataset=Cifar100 \
--dataset_path=/home/zhenyulin/Training_data \
--pruning_mode=Layerwise \
--pruning_method=K-L1norm \
--calculate_k=Cifar100_K \
&& \
python3 Resnet18_pruning.py \
--weight_path=Weights/Cifar100/resnet18/acc68.35%_resnet18 \
--dataset=Cifar100 \
--dataset_path=/home/zhenyulin/Training_data \
--pruning_mode=Layerwise \
--pruning_method=K-Distance \
--calculate_k=Cifar100_K \
&& \
python3 Resnet18_pruning.py \
--weight_path=Weights/Cifar100/resnet18/acc68.35%_resnet18 \
--dataset=Cifar100 \
--dataset_path=/home/zhenyulin/Training_data \
--pruning_mode=Layerwise \
--pruning_method=K-Taylor \
--calculate_k=Cifar100_K \
&& \
python3 Resnet18_pruning.py \
--weight_path=Weights/Cifar100/resnet18/acc68.35%_resnet18 \
--dataset=Cifar100 \
--dataset_path=/home/zhenyulin/Training_data \
--pruning_mode=Layerwise \
--pruning_method=Taylor \
--calculate_k=Cifar100_K \
&& \
python3 Resnet18_pruning.py \
--weight_path=Weights/Cifar100/resnet18/acc68.35%_resnet18 \
--dataset=Cifar100 \
--dataset_path=/home/zhenyulin/Training_data \
--pruning_mode=Layerwise \
--pruning_method=L1norm \
--calculate_k=Cifar100_K \
&& \
python3 Resnet18_pruning.py \
--weight_path=Weights/Cifar10/resnet18/acc91.82%_resnet18 \
--dataset=Cifar10 \
--dataset_path=/home/zhenyulin/Training_data \
--pruning_mode=Layerwise \
--pruning_method=K-L1norm \
--calculate_k=Cifar10_K \
&& \
python3 Resnet18_pruning.py \
--weight_path=Weights/Cifar10/resnet18/acc91.82%_resnet18 \
--dataset=Cifar10 \
--dataset_path=/home/zhenyulin/Training_data \
--pruning_mode=Layerwise \
--pruning_method=K-Distance \
--calculate_k=Cifar10_K \
&& \
python3 Resnet18_pruning.py \
--weight_path=Weights/Cifar10/resnet18/acc91.82%_resnet18 \
--dataset=Cifar10 \
--dataset_path=/home/zhenyulin/Training_data \
--pruning_mode=Layerwise \
--pruning_method=K-Taylor \
--calculate_k=Cifar10_K \
&& \
python3 Resnet18_pruning.py \
--weight_path=Weights/Cifar10/resnet18/acc91.82%_resnet18 \
--dataset=Cifar10 \
--dataset_path=/home/zhenyulin/Training_data \
--pruning_mode=Layerwise \
--pruning_method=Taylor \
--calculate_k=Cifar10_K \
&& \
python3 Resnet18_pruning.py \
--weight_path=Weights/Cifar10/resnet18/acc91.82%_resnet18 \
--dataset=Cifar10 \
--dataset_path=/home/zhenyulin/Training_data \
--pruning_mode=Layerwise \
--pruning_method=L1norm \
--calculate_k=Cifar10_K \
