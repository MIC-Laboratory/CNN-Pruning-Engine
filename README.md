# Pytorch-CNN-Pruning
## Environment Setup:
```conda create --name Pytorch --file conda-environment.yml```

## Platforms
- Ubuntu 18.04
- Pytorch=1.10.1
- py3.9
- cuda10.2
- cudnn7.6.5_0


## TensorBoard Commend:
```tensorboard --logdir [Directory]```

If you obtain PORT problem, can use the commend below to change ports.


```--port = [PORT]```


## MobilenetV2 Pruning Result

Labels:

![MobilenetV2-Data-Labels](MobilenetV2-data/Labels.png)

ACC:

![MobilenetV2-Data-ACC](MobilenetV2-data/ACC.svg)

Flops:

![MobilenetV2-Data-MACs](MobilenetV2-data/MACs(G).svg)

Params:

![MobilenetV2-Data-Params](MobilenetV2-data/Params(M).svg)
