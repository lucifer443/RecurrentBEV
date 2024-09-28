# Train & inference

#### 1. Train

You can train the model following:

```bash
./tools/dist_train.sh configs/recurrentbev/recurrentbev_res50_704x256_ep90.py 8 --work-dir work_dirs/recurrentbev_res50_704x256_ep90/
```

**Notes**: 

- The results reported in our paper is all trained with total batch size 64, while consumes huge GPU memory and A100-80G is needed.

#### 2. Evaluation

You can evaluate the detection model following:

```bash
./tools/dist_test.sh configs/recurrentbev/recurrentbev_res50_704x256_ep90.py work_dirs/recurrentbev_res50_704x256_ep90/latest.pth 8
```

### Training Tips

We conduct all experiments using A100-80G with total batch size 64. Considering that such experimental conditions are not available to all researchers，we provide some tips for saving GPU memory usage here.

* Change the type of voxel pooling from "bevfusion" to "bev_pool_v2". For example 
  
  ```python
  # from "configs/recurrentbev/_base_/recurrent_bev.py
  # 89~94
        view_trans=dict(
            type='LiftSplatVT',
            grid_config=bev_grid_config,
            input_size=(256, 704),
            downsample=16,
            pool_method='bev_pool_v2'),  # bevfusion -> bev_pool_v2
  ```

* Use  automatic-mixed-precision (AMP) in training process.
  
  ```bash
  ./tools/dist_train.sh configs/recurrentbev/recurrentbev_res50_704x256_ep90.py 8 --work-dir work_dirs/recurrentbev_res50_704x256_ep90/ --amp
  ```

* Using activation checkpointing to save memory.
  
  ```python
  # from "configs/recurrentbev/_base_/recurrent_bev.py
  # 95~98
          post_process=dict(
              inplanes=C_img,
              planes=C_img,
              num_blocks=2,
              with_cp=True), # using activation checkpointing
  
  # 106~119
      bev_backbone=dict(
          type='CustomResNet',
          depth=18,
          with_stem=True,
          with_maxpool_in_stem=False,
          in_channels=C_img * 2,
          stem_channels=C_img * 2,
          base_channels=C_img * 2,
          num_stages=3,
          strides=(1, 2, 2),
          dilations=(1, 1, 1),
          out_input=False,
          out_indices=(0, 2),
          norm_eval=False,
          with_cp=True),
  ```
