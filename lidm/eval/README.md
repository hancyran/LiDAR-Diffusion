# Evaluation Toolbox for LiDAR Generation

This directory is a **self-contained**, **memory-friendly** and mostly **CUDA-accelerated** toolbox of multiple evaluation metrics for LiDAR generative models, including:
* Perceptual metrics (our proposed):
  * Fréchet Range Image Distance (**FRID**)
  * Fréchet Sparse Volume Distance (**FSVD**)
  * Fréchet Point-based Volume Distance (**FPVD**)
* Statistical metrics (proposed in [Learning Representations and Generative Models for 3D Point Clouds](https://arxiv.org/abs/1707.02392)):
  * Minimum Matching Distance (**MMD**)
  * Jensen-Shannon Divergence (**JSD**)
* Statistical pairwise metrics (for reconstruction only):
  * Chamfer Distance (**CD**)
  * Earth Mover's Distance (**EMD**)

## Citation

If you find this project useful in your research, please consider citing:
```
@article{ran2024towards,
  title={Towards Realistic Scene Generation with LiDAR Diffusion Models},
  author={Ran, Haoxi and Guizilini, Vitor and Wang, Yue},
  journal={arXiv preprint arXiv:2404.00815},
  year={2024}
}
```


## Dependencies

### Basic (install through **pip**):
* scipy
* numpy
* torch
* pyyaml

### Required by FSVD and FPVD:
* [Torchsparse v1.4.0](https://github.com/mit-han-lab/torchsparse/tree/v1.4.0) (pip install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0)
* [Google Sparse Hash library](https://github.com/sparsehash/sparsehash) (apt-get install libsparsehash-dev **or** compile locally and update variable CPLUS_INCLUDE_PATH with directory path)


## Model Zoo 

To evaluate with perceptual metrics on different types of LiDAR data, you can download all models through:
*  this [google drive link](https://drive.google.com/file/d/1Ml4p4_nMlwLkSp7JB528GJv2_HxO8v1i/view?usp=drive_link) in the .zip file 

or
*  the **full directory** of one specific model:

### 64-beam LiDAR (trained on [SemanticKITTI](http://semantic-kitti.org/dataset.html)):

| Metric |                                            Model                                            |          Arch           |                                                  Link                                                   | Code                                                             | Comments                                                                  |
|:------:|:-------------------------------------------------------------------------------------------:|:-----------------------:|:-------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------|---------------------------------------------------------------------------|
|  FRID  | [RangeNet++](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf) |  DarkNet21-based UNet   | [Google Drive](https://drive.google.com/drive/folders/1ZS8KOoxB9hjB6kwKbH5Zfc8O5qJlKsbl?usp=drive_link) | [./models/rangenet/model.py](./models/rangenet/model.py)         | range image input (our trained model without the need of remission input) |
|  FSVD  |                      [MinkowskiNet](https://arxiv.org/abs/1904.08755)                       |       Sparse UNet       | [Google Drive](https://drive.google.com/drive/folders/1zN12ZEvjIvo4PCjAsncgC22yvtRrCCMe?usp=drive_link) | [./models/minkowskinet/model.py](./models/minkowskinet/model.py) | point cloud input                                                         |
|  FPVD  |                         [SPVCNN](https://arxiv.org/abs/2007.16100)                          | Point-Voxel Sparse UNet | [Google Drive](https://drive.google.com/drive/folders/1oEm3qpxfGetiVAfXIvecawEiFqW79M6B?usp=drive_link) | [./models/spvcnn/model.py](./models/spvcnn/model.py)             | point cloud input                                                         |


### 32-beam LiDAR (trained on [nuScenes](https://www.nuscenes.org/nuscenes)):

| Metric |                      Model                       |          Arch           |                                                  Link                                                   | Code                                                             | Comments          |
|:------:|:------------------------------------------------:|:-----------------------:|:-------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------|-------------------|
|  FSVD  | [MinkowskiNet](https://arxiv.org/abs/1904.08755) |       Sparse UNet       | [Google Drive](https://drive.google.com/drive/folders/1oZIS9FlklCQ6dlh3TZ8Junir7QwgT-Me?usp=drive_link) | [./models/minkowskinet/model.py](./models/minkowskinet/model.py) | point cloud input |
|  FPVD  |    [SPVCNN](https://arxiv.org/abs/2007.16100)    | Point-Voxel Sparse UNet | [Google Drive](https://drive.google.com/drive/folders/1F69RbprAoT6MOJ7iI0KHjxuq-tbeqGiR?usp=drive_link) | [./models/spvcnn/model.py](./models/spvcnn/model.py)             | point cloud input |


## Usage

1. Place the unzipped `pretrained_weights` folder under the root python directory **or** modify the `DEFAULT_ROOT` variable in the `__init__.py`.
2. Prepare input data, including the synthesized samples and the reference dataset. **Note**: The reference data should be the **point clouds projected back from range images** instead of raw point clouds. 
3. Specify the data type (`32` or `64`) and the metrics to evaluate. Options: `mmd`, `jsd`, `frid`, `fsvd`, `fpvd`, `cd`, `emd`.
4. (Optional) If you want to compute `frid`, `fsvd` or `fpvd` metric, adjust the corresponding batch size through the `MODAL2BATCHSIZE` in file `__init__.py` according to your max GPU memory (default: ~24GB).
5. Start evaluation and all results will print out!

### Example:

```
from .eval_utils import evaluate

data = '64'  # specify data type to evaluate
metrics = ['mmd', 'jsd', 'frid', 'fsvd', 'fpvd']  # specify metrics to evaluate

# list of np.float32 array
# shape of each array: (#points, #dim=3), #dim: xyz coordinate (NOTE: no need to input remission)
reference = ...
samples = ...

evaluate(reference, samples, metrics, data)
```


## Acknowledgement

- The implementation of MinkowskiNet and SPVCNN is borrowed from [2DPASS](https://github.com/yanx27/2DPASS).
- The implementation of RangeNet++ is borrowed from [the official RangeNet++ codebase](https://github.com/PRBonn/lidar-bonnetal).
- The implementation of Chamfer Distance is adapted from [CD Pytorch Implementation](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) and Earth Mover's Distance from [MSN official repo](https://github.com/Colin97/MSN-Point-Cloud-Completion).
