# Directiory
    ├── Imagenet_K
    │   ├── Cifar10
    │   │   ├── K-Distance
    │   │   ├── K-L1norm
    │   │   ├── K-Taylor
    │   │   ├── L1norm
    │   │   └── Taylor
    │   ├── Cifar100
    │   │   ├── K-Distance
    │   │   ├── K-L1norm
    │   │   ├── K-Taylor
    │   │   ├── L1norm
    │   │   └── Taylor
    │   └── Imagenet
    │       ├── K-Distance
    │       ├── K-L1norm
    │       ├── K-Taylor
    │       ├── L1norm
    │       └── Taylor
    ├── K_Selection
    │   ├── Cifar10
    │   │   └── K-L1norm
    │   ├── Cifar100
    │   │   └── vgg16
    │   └── Imagenet
    │       └── Vgg
    └── Own_K
        ├── Cifar10
        │   ├── K-Distance
        │   ├── K-L1norm
        │   ├── K-Taylor
        │   ├── L1norm
        │   └── Taylor
        └── Cifar100
            ├── K-Distance
            ├── K-L1norm
            ├── K-Taylor
            ├── L1norm
            └── Taylor

# Directory Descirption
| Folder Name | Description |
| ----------- | ----------- |
| Imagenet_K  | Testing Result of Full Layer Pruning in imagenet dataset by using the K value set that calulated using imagenet dataset |
| K_Selection | Test different K values (from 1 to n/2 [n = # of filters in layer]) in different dataset |
| Own_K | Testing Result of Full Layer Pruning in different dataset by using the K value set that calulated using the corrspending dataset

