# Data Preparation

#### 1. Download nuScenes

Download the [nuScenes dataset](https://www.nuscenes.org/download) to `./data/nuscenes`.

#### 2. Download annotations

Download the [annotation files](https://drive.google.com/drive/folders/1s69G7LhkB0PBk38S1J_1iyVkx5HLsMnh?usp=sharing) to `./data/nuscenes/annotations`.

#### 3. Download pretrained weights

Using pre-trained weights from the external dataset i.e.nuImages needs to download to ./checkpoints .

### Folder structure

After preparation, you will be able to see the following directory structure:  

```
StreamPETR
├── bev/
├── configs/
├── docker/
├── docs/
├── resources/
├── tools/
├── checkpoints/
│   ├── cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth
│   ├── dd3d_det_final.pth
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── annotations/
|   |   |   ├── nuscenes_temporal_infos_train.pkl
|   |   |   ├── nuscenes_temporal_infos_trainval.pkl
|   |   |   ├── nuscenes_temporal_infos_val.pkl
|   |   |   ├── nuscenes_temporal_infos_test.pkl
```
