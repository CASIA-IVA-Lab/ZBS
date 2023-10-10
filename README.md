**ZBS**: Zero-shot Background Subtraction via Instance-level Background Modeling and Foreground Selection
========

## Introduction

This repository is an official implementation of the **[ZBS](https://arxiv.org/abs/2303.14679)**.
ZBS fully utilizes the advantages of zero-shot object detection to build the open-vocabulary instance-level background model.
It can detect most of the categories in the real world and can detect the unseen foreground categories outside the pre-defined categories. ZBS
achieves remarkably **4.70%** F-Measure improvements over state-of-the-art unsupervised methods.

<p align="center"> <img src='docs/arch.svg' align="center" height="600px"> </p>        

## Features

- The first unsupervised zero-shot background subtraction.

- The first background subtraction method based on an instance-level background model.

- Detects **any** class given class names (using [Detic](https://github.com/facebookresearch/Detic)).

- State-of-the-art results on CDnet 2014 dataset compared with other unsupervised background subtraction method.

## Instructions

See [GET_STARTED.md](docs/GET_STARTED.md).

## Main Results

Overall and per-category F-Measure comparison of different Unsupervised BGS methods on the CDnet 2014 dataset.

| Unsupervised BGS | baseline | camjitt | dynbg  | intmot | shadow | thermal | badwea | lowfr | night  | PTZ   | turbul | Overall |
|:-----------:|:--------:|:-------:|:------:|:------:|:------:|:-------:|:------:|:-----:|:------:|:-----:|:------:|:-------:|
| PAWCS       | 0.9397   | 0.8137  | 0.8938 | 0.7764 | 0.8913 | 0.8324  | 0.8152 | 0.6588| 0.4152 | 0.4615| 0.6450 | 0.7403  |
| SuBSENSE    | 0.9503   | 0.8152  | 0.8177 | 0.6569 | 0.8986 | 0.8171  | 0.8619 | 0.6445| 0.5599 | 0.3476| 0.7792 | 0.7408  |
| WisenetMD   | 0.9487   | 0.8228  | 0.8376 | 0.7264 | 0.8984 | 0.8152  | 0.8616 | 0.6404| 0.5701 | 0.3367| **0.8304** | 0.7535  |
| SWCD        | 0.9214   | 0.7411  | 0.8645 | 0.7092 | 0.8779 | 0.8581  | 0.8233 | 0.7374| 0.5807 | 0.4545| 0.7735 | 0.7583  |
| SemanticBGS | 0.9604   | 0.8388  | **0.9489** | 0.7878 | 0.9478 | 0.8219  | 0.8260 | **0.7888**| 0.5014 | 0.5673| 0.6921 | 0.7892  |
| RTSS        | 0.9597   | 0.8396  | 0.9325 | 0.7864 | 0.9551 | 0.8510  | 0.8662 | 0.6771| 0.5295 | 0.5489| 0.7630 | 0.7917  |
| RT-SBS-v2   | 0.9535   | 0.8233  | 0.9217 | **0.8946** | 0.9497 | 0.8697  | 0.8279 | 0.7341| 0.5629 | 0.5808| 0.7315 | 0.8045  |
| ZBS (Ours)  | **0.9653**   | **0.9545**  | 0.9290 | 0.8758 | **0.9765** | **0.8698**  | **0.9229** | 0.7433| **0.6800** | **0.8133**| 0.6358 | **0.8515**  |

Overall and per-category result of ZBS on the CDnet 2014 dataset.

| Category  | Recall | Specificity | PWC   | Precision | F-Measure |
|:---------:|:------:|:-----------:|:-----:|:---------:|:---------:|
| badWea    | 0.9049 |    0.9988   | 0.2755|   0.9439  |   0.9229  |
| baseline  | 0.9709 |    0.9988   | 0.2237|   0.9603  |   0.9653  |
| camjitt   | 0.9543 |    0.9979   | 0.4022|   0.9554  |   0.9545  |
| dynbg     | 0.9269 |    0.9996   | 0.0951|   0.9340  |   0.9290  |
| intmot    | 0.8254 |    0.9965   | 1.6864|   0.9481  |   0.8758  |
| lowfr     | 0.7302 |    0.9988   | 0.3279|   0.7584  |   0.7433  |
| night     | 0.6341 |    0.9958   | 1.2477|   0.7666  |   0.6800  |
| PTZ       | 0.7490 |    0.9997   | 0.2387|   0.9223  |   0.8133  |
| shadow    | 0.9712 |    0.9991   | 0.2097|   0.9819  |   0.9765  |
| thermal   | 0.8475 |    0.9954   | 1.1686|   0.9040  |   0.8698  |
| turbul    | 0.7286 |    0.9984   | 0.3198|   0.6075  |   0.6358  |
| Overall   | 0.8403 |    0.9981   | 0.5632|   0.8802  |   0.8515  |


## Citation

If you find this project useful for your research, please consider citing this paper.

```
@inproceedings{
an2023zbs,
title={{ZBS}: Zero-shot Background Subtraction via instance-level background modeling and foreground selection},
author={Yongqi An and Xu Zhao and Tao Yu and Haiyun Guo and Chaoyang Zhao and Ming Tang and Jinqiao Wang},
booktitle={Conference on Computer Vision and Pattern Recognition 2023},
year={2023},
url={https://openreview.net/forum?id=f-9UZN4GEV}
}
```

## Acknowledgement

Our repository is mainly built upon [Detic](https://github.com/facebookresearch/Detic).
