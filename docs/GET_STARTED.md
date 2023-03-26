**ZBS**: Zero-shot Background Subtraction via Instance-level Background Modeling and Foreground Selection
========

## Installation
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.8.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

### Example conda environment setup
First, create a new conda environment. We suggest you to install pytorch 1.8.
```bash
conda create --name zbs python=3.8 -y
conda activate zbs
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```
Then, clone the repository locally and install dependencies:
```bash
# under your working directory
git clone git@github.com:CASIA-IVA-Lab/ZBS.git
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ../ZBS
pip install -r requirements.txt
```
Last, download the pretrained model, you can get more to check Detic's [MODEL ZOO](docs/MODEL_ZOO.md).
```bash
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

## Data Preparation

Our experiments use [CDnet 2014](https://www.kaggle.com/datasets/fc82cc044b7e90db502e947e3a4d301a0ff2c498a38b75522543304a40c764f5?resource=download-directory) and [ABODA](https://github.com/kevinlin311tw/ABODA).
Before starting processing, please download the (selected) datasets from the official websites and place or sim-link them under `$ZBS_ROOT/datasets/`. 

```
$Detic_ROOT/datasets/
    metadata/
    cdnet2014/
    custom_video/
    ABODA/
```
`metadata/` is our preprocessed meta-data (included in the repo). See the below [section](#Metadata) for details.
Please follow the following instruction to pre-process individual datasets.

### CDnet 2014 dataset
This dataset contains 11 video categories with 4 to 6 videos sequences in each category.

Download CDnet 2014 dataset from the website. We only need the  dataset in this project:
```
cdnet2014/
    PTZ/
        continuousPan/
            groundtruth/
            ...
            temporalROI.txt/
        ...
        zoomInZoomOut/
            groundtruth/
            ...
            temporalROI.txt/
    ...
    turbulence/
        ...

```

### ABODA

ABandoned Objects DAtaset (ABODA) is a new public dataset for abandoned object detection. ABODA comprises 11 sequences labeled with various real-application scenarios that are challenging for abandoned-object detection.

Download ABODA from the website:

```
ABODA/
    video1.avi
    ...
    video11.avi
```
### Metadata

```
metadata/
    lvis_v1_train_cat_info.json
    coco_clip_a+cname.npy
    lvis_v1_clip_a+cname.npy
    o365_clip_a+cnamefix.npy
    oid_clip_a+cname.npy
    imagenet_lvis_wnid.txt
    Objects365_names_fix.csv
```

`lvis_v1_train_cat_info.json` is used by the Federated loss.
This is created by 
```bash
python tools/get_lvis_cat_info.py --ann datasets/lvis/lvis_v1_train.json
```

`*_clip_a+cname.npy` is the pre-computed CLIP embeddings for each datasets.
They are created by (taking LVIS as an example)
```bash
python tools/dump_clip_features.py --ann datasets/lvis/lvis_v1_val.json --out_path metadata/lvis_v1_clip_a+cname.npy
```
Note we do not include the 21K class embeddings due to the large file size.
To create it, run
```bash
python tools/dump_clip_features.py --ann datasets/lvis/lvis_v1_val_lvis-21k.json --out_path datasets/metadata/lvis-21k_clip_a+cname.npy
```

`imagenet_lvis_wnid.txt` is the list of matched classes between ImageNet-21K and LVIS.

`Objects365_names_fix.csv` is our manual fix of the Objects365 names.

## Demo

### Demo for CDnet 2014 dataset
To get the results on CDnet 2014 on a single GPU:
```bash
bash script/demo.sh cdnet
```

### Demo for CDnet 2014 dataset
To evaluate the performance of ZBS on CDnet 2014 on a single GPU:
```bash
bash script/demo.sh test
```

### Demo for ABODA
To evaluate ZBS on video on a single GPU:
```bash
bash script/demo.sh video datasets/ABODA/video1.avi 
```
