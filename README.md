# CAT-KD
This repository is the official implementation of our paper '[Class Attention Transfer Based Knowledge Distillation](https://arxiv.org/abs/2304.12777)', accepted in CVPR 2023.
# Guidance
Our implementation is based on [MDistiller](https://github.com/megvii-research/mdistiller). Here we introduce the guidance for reproducing the experiments reported in the paper, more detailed usage of the framework please refer to [MDistiller](https://github.com/megvii-research/mdistiller).

## Preparation
1. Download [ImageNet](https://image-net.org/) and move them to CAT-KD/data/imagenet.
2. Download [pre-trained teachers](https://github.com/megvii-research/mdistiller/releases/tag/checkpoints) and untar them to CAT-KD/download_ckpts/cifar_teachers.

## Reproduction
All reported experiments can be easily reproduced by selecting/modifying our preset configuration file.
``` python
# CAT-KD.
python tools/train.py --cfg configs/cifar100/CAT_KD/res32x4_shuv1.yaml

# CAT, where the transferred CAMs are binarized.
python tools/train.py --cfg configs/cifar100/CAT/CAT_Binarization/res32x4_res32x4.yaml
```
To facilitate the reproduction/exploration of CAT/CAT-KD, here we present the function of the keywords contained in the config files.
| Keyword                    | Function                                                                                                 |
|----------------------------|----------------------------------------------------------------------------------------------------------|
| CAT_loss_weight            | β, the coefficient of CAT loss.                                                 |
| CAM_RESOLUTION             | The resolution of the transferred CAMs after the average pooling function.                               |
| onlyCAT                    | True: Only CAT loss is used.<br>False: Both CAT loss and CE loss are used. |
| IF_NORMALIZE               | True: perform normalization on the transferred CAMs.                                                     |
| IF_BINARIZE                | True: perform binarization on the transferred CAMs.                                                      |
| REDUCTION                  | True: perform data reduction on the training set of CIFAR-100.                                           |
| RESERVED_CLASS_NUM         | Number of categories of the training set data after the reduction.                                       |
| RESERVED_RATE              | Training set data reserved rate (per class).                                                             |
| IF_OnlyTransferPartialCAMs | True: only transfer CAMs of the certain classes.                                                         |
| CAMs_Nums                  | number of classes of the transferred CAMs                                                                |
| Strategy                   | 0: select CAMs with top n score.<br>1: select CAMs with the lowest n scores.                                |
# Notation
## Don't forget to tune β during your exploration with CAT.
1. Since our experiments with CAT are mianly conducted to explore the properties of transferring CAMs, we have not tuned β to improve its performance.
2. The value of CAT loss is affected by the models' architecture, don't forget to tune β to keep CAT loss within a reasonable range (maybe 1~50 after the first epoch).
## Usage of the distillation framework.
Our implementation is based on [MDistiller](https://github.com/megvii-research/mdistiller), which is an efficient distillation framework. If what you need is a framework to implement your method, we recommend you to use the vanilla version.
# Citation
Please cite our paper if our paper/code helps your research.
```
@inproceedings{guo2023class,
  title={Class Attention Transfer Based Knowledge Distillation},
  author={Guo, Ziyao and Yan, Haonan and Li, Hui and Lin, Xiaodong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11868--11877},
  year={2023}
}
```
