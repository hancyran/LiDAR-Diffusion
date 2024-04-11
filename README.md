<div align="center">
<h1>LiDAR Diffusion Models [CVPR 2024]</h1>

[**Haoxi Ran**](https://hancyran.github.io/) · [**Vitor Guizilini**](https://scholar.google.com.br/citations?user=UH9tP6QAAAAJ&hl=en) · [**Yue Wang**](https://yuewang.xyz/)


<a href="https://arxiv.org/abs/2404.00815"><img src='https://img.shields.io/badge/arXiv-LiDAR Diffusion-red' alt='arXiv'></a>
<a href="https://arxiv.org/pdf/2404.00815.pdf"><img src='https://img.shields.io/badge/PDF-LiDAR Diffusion-Green' alt='PDF'></a>
<a href="#citation"><img src='https://img.shields.io/badge/BibTex-LiDAR Diffusion-blue' alt='Paper BibTex'></a>

<img src=assets/overview.png width="400"/>

</div>

## News

- [**Apr 5, 2024**] Most code and a detailed study of our autoencoder design along with the pretrained models is released! 



## Requirements

We provide an available [conda](https://conda.io/) environment named `lidm`:

```
sh init/create_env.sh
conda activate lidm
```

## Evaluation Toolbox

**Overview of evaluation metrics**:

<table align="center">
<thead>
  <tr>
    <th style="text-align: center; vertical-align: middle;" colspan="3">Perceptual Metrics<br>(generation &amp; reconstruction)</th>
    <th style="text-align: center; vertical-align: middle;" colspan="2">Statistical Metrics<br>(generation only)</th>
    <th style="text-align: center; vertical-align: middle;" colspan="2">Distance metrics <br> (reconstruction only)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="text-align: center; vertical-align: middle;">FRID</td>
    <td style="text-align: center; vertical-align: middle;">FSVD</td>
    <td style="text-align: center; vertical-align: middle;">FPVD</td>
    <td style="text-align: center; vertical-align: middle;">JSD</td>
    <td style="text-align: center; vertical-align: middle;">MMD</td>
    <td style="text-align: center; vertical-align: middle;">CD</td>
    <td style="text-align: center; vertical-align: middle;">EMD</td>
  </tr>
</tbody>
</table>
<br/>

To standardize the evaluation of LiDAR generative models, we provide a **self-contained** and **mostly CUDA-accelerated** evaluation toolbox in the directory `./lidm/eval/`. It implements and integrates various evaluation metrics, including:
* Perceptual metrics:
  * Fréchet Range Image Distance (**FRID**)
  * Fréchet Sparse Volume Distance (**FSVD**)
  * Fréchet Point-based Volume Distance (**FPVD**)
* Statistical metrics:
  * Minimum Matching Distance (**MMD**)
  * Jensen-Shannon Divergence (**JSD**)
* Statistical pairwise metrics (for reconstruction only):
  * Chamfer Distance (**CD**)
  * Earth Mover's Distance (**EMD**)



For more details about setup and usage, please refer to the [Evaluation Toolbox README](./lidm/eval/README.md).


## Model Zoo 

To test different tasks below, please download the pretrained LiDM and its corresponding autoencoder:

### Pretrained Autoencoders

Coming Soon...

### Pretrained LiDMs

Coming Soon...

### Study on Design of Autoencoders 

All the following experiments are conducted with 8 NVIDIA 3090 GPUs and _40k training steps_ on KITTI-360 (64-beam).

Tip: Download the video instead of watching it with the Google Drive's built-in video player provides a better visualization.

| Curvewise <br/> Factor | Patchwise <br/> Factor | Output <br/> Size | rFRID(↓) | rFSVD(↓) | rFPVD(↓) | CD(↓) | EMD(↓) | #Params (M) |                                                Directory                                                |                       Rec.&nbsp;Results&nbsp;val<br/>(Range&nbsp;Image)                        |                       Rec.&nbsp;Results&nbsp;val<br/>(Point&nbsp;Cloud)                        |
|:----------------------:|:----------------------:|:-----------------:|:--------:|:--------:|:--------:|:-----:|:------:|:-----------:|:-------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|
|          N/A           |          N/A           |   Ground Truth    |    -     |    -     |    -     |   -   |   -    |      -      |                                                    -                                                    |  [Video](https://drive.google.com/file/d/1wAtQSlVwF2jCpcL3zbXlk2lGUYzo1GBf/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/1iHIB7Jw-WS0D_hXgQSOyyDyWCmPVR-6k/view?usp=sharing)   |
|                        |                        |                   |          |          |          |       |        |             |                                                                                                         |                                                                                                |                                                                                                |
|           4            |           1            |     64x256x2      |   0.2    |   12.9   |   13.8   | 0.069 | 0.151  |    9.52     | [Google Drive](https://drive.google.com/drive/folders/1bLGigdh3oNBTfskdX5yisqJ3fd99wFnR?usp=drive_link) | [Video](https://drive.google.com/file/d/1w7slbsRjlU4kb0kl6LyjX-JojJvoWQhG/view?usp=drive_link) | [Video](https://drive.google.com/file/d/17ewPXoRMeA_HsvEOznsvxy3d6iKk7hC2/view?usp=drive_link) |
|           8            |           1            |     64x128x3      |   0.9    |   21.2   |   17.4   | 0.141 | 0.230  |    10.76    |  [Google Drive](https://drive.google.com/drive/folders/1qPCPJC9TsIEO2UaZqurPu99m4syzfzuq?usp=sharing)   |  [Video](https://drive.google.com/file/d/17kukYFlJY40_cVBuWXMLHiMe7ls2OLNh/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/116IXDMgrWn6OHtyEYIo6aM1ARloX3BWF/view?usp=sharing)   |
|           16           |           1            |      64x64x4      |   2.8    |   31.1   |   23.9   | 0.220 | 0.265  |    12.43    |  [Google Drive](https://drive.google.com/drive/folders/1IHm3KlwG4lQAa9Ygt3WRUPfDxAQ1Tjia?usp=sharing)   |  [Video](https://drive.google.com/file/d/12TKyoajTiU_hr1MAdK2PNveddorCshG4/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/18NCV7JoR3W1COaPH96a1ozbh8-58eT6n/view?usp=sharing)   |
|           32           |           1            |      64x32x8      |   16.4   |   49.0   |   38.5   | 0.438 | 0.344  |    13.72    |  [Google Drive](https://drive.google.com/drive/folders/1CnUGOoAZDrSbDG3DjVx5pcouAT5WQTGN?usp=sharing)   |  [Video](https://drive.google.com/file/d/1S2DPHfWAljKZrHJlPHIvxAPK2-rpdJ_J/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/1yx8V4Qav7sCigcfSHrrrJQOFF-s2PryV/view?usp=sharing)   |
|                        |                        |                   |          |          |          |       |        |             |                                                                                                         |                                                                                                |                                                                                                |
|           1            |           2            |     32x512x2      |   1.5    |   25.0   |   23.8   | 0.096 | 0.178  |    2.87     |  [Google Drive](https://drive.google.com/drive/folders/16OLfvexGSuOO8zNxkVLvY6rglvLn3HRG?usp=sharing)   |  [Video](https://drive.google.com/file/d/1tPPD2Pnn_6ge3x2yoJXhkDhe0Wi5Qxhw/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/1Xjg0ckVb208BFEgbv4VQtV-fVraEXUNC/view?usp=sharing)   |
|           1            |           4            |     16x256x4      |   0.6    |   15.4   |   15.8   | 0.142 | 0.233  |    12.45    |  [Google Drive](https://drive.google.com/drive/folders/1ArTAar3UM-7eBmkGb2bqDF0MVW6GL0az?usp=sharing)   |  [Video](https://drive.google.com/file/d/1Q_ZTRKyDOAmP314p9B6Cip79mc-FJ2se/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/1-t9zvSrov1OsF_WEIBqH3xkLzTJfxRBr/view?usp=sharing)   |
|           1            |           8            |     8x128x16      |   17.7   |   35.7   |   33.1   | 0.384 | 0.327  |    15.78    |  [Google Drive](https://drive.google.com/drive/folders/1Ol2P6ZYYFjEImLAhIhY8iR_G6bLKI4Yx?usp=sharing)   |  [Video](https://drive.google.com/file/d/14hPy2utsaxwPxW5PA7gO7ak7f-lcd-X5/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/1izj-_1hFkdaRCg2qUzkXByfCD-vBd_1M/view?usp=sharing)   |
|           1            |           16           |      4x64x64      |   37.1   |   68.7   |   63.9   | 0.699 | 0.416  |    16.25    |  [Google Drive](https://drive.google.com/drive/folders/1_vihPf9xgnr4Zib-dYNUZ1n6kTMxT3rG?usp=sharing)   |  [Video](https://drive.google.com/file/d/1G7evMm3H6WvbHFhBlCa8wxPzwVC3q-8H/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/1IdBrEpCIugvxVHyNOsNIg8Y8ZBWrHcWL/view?usp=sharing)   |
|                        |                        |                   |          |          |          |       |        |             |                                                                                                         |                                                                                                |                                                                                                |
|           2            |           2            |     32x256x3      |   0.4    |   11.2   |   12.2   | 0.094 | 0.199  |    13.09    |  [Google Drive](https://drive.google.com/drive/folders/1SdFEtMGRE9Oi23jlDrtebslc5hxhYLBQ?usp=sharing)   |  [Video](https://drive.google.com/file/d/1Ac4jVB6RkqMwV1fZcPGDyQhR3eE_Zj6C/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/1pg2ezSmXiu3ensvj564JIy6CpB46uZm7/view?usp=sharing)   |
|           4            |           2            |     32x128x4      |   3.9    |   19.6   |   16.6   | 0.197 | 0.236  |    14.35    |  [Google Drive](https://drive.google.com/drive/folders/1uWlZPiU9Jw4TFfvI4Avi4r0bEyJ9kw4i?usp=sharing)   |  [Video](https://drive.google.com/file/d/1yZGqe_DcDXew3JabnN4T1-P27ZlscHba/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/1i_q6gVY4gMtzKYlhlMQ9QrRql73VX05j/view?usp=sharing)   |
|           8            |           2            |      32x64x8      |   8.0    |   25.3   |   20.2   | 0.277 | 0.294  |    16.06    |  [Google Drive](https://drive.google.com/drive/folders/1Z9B7PjR5SlgAl2WLGmIPxiYTzmo17J--?usp=sharing)   |  [Video](https://drive.google.com/file/d/1HVqFbIE1lgotDplc8x7_hJkSU5vLtbRN/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/1jSYWZMmPelmfWpVa7V5f2Byr9vN2BKXo/view?usp=sharing)   | 
|           16           |           2            |     32x32x16      |   21.5   |   54.2   |   44.6   | 0.491 | 0.371  |    17.44    |  [Google Drive](https://drive.google.com/drive/folders/1jBaEiAymHACWTdy_GbYOiG9e-GFVkIfe?usp=sharing)   |  [Video](https://drive.google.com/file/d/1flAzjRLcl5Jtc_T--GbbomKWi42DvW9v/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/1zfMzu6NFeLJhR1YPU28k7vPy1GX-80QT/view?usp=sharing)   |
|           2            |           4            |     16x128x8      |   2.5    |   16.9   |   15.8   | 0.205 | 0.273  |    15.07    |  [Google Drive](https://drive.google.com/drive/folders/1w-4bF4yORsot6xb5ia95RXWhfHrfpK0T?usp=sharing)   |  [Video](https://drive.google.com/file/d/1rm0sviRg4LfImgWVCi6THi3pHF4kFccH/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/1gPKB2zj44oLLEBuUXU8uiaXIcSWpyMOi/view?usp=sharing)   |
|           4            |           4            |     16x128x16     |   13.8   |   29.5   |   25.4   | 0.341 | 0.317  |    16.86    |  [Google Drive](https://drive.google.com/drive/folders/1_hY52mbKy4t3U5eWQ4Stq-3wZX1FPXXz?usp=sharing)   |  [Video](https://drive.google.com/file/d/1ldMRXfUtFNBtjCCc-KYR311dQvCmn0EF/view?usp=sharing)   |  [Video](https://drive.google.com/file/d/129WcZXW3b6e4UMxZ9x4XCR3BlaKw1Vec/view?usp=sharing)   |


## Unconditional LiDAR Generation

<p align="center">
<img src=assets/uncond.jpeg width="512"/>
</p>

To run sampling on pretrained models (and to evaluate your results with flag "--eval"):
```
CUDA_VISIBLE_DEVICES=0 python scripts/sample.py -r models/lidm/kitti/uncond/model.ckpt -d kitti [--eval]
```


## Semantic-Map-to-LiDAR

<p align="center">
<img src=assets/map2lidar.gif width="768"/>
</p>

To check the conditional results on a full sequence of semantic maps (sequence '08'), please refer to [this video](https://drive.google.com/file/d/1TtAROAmQVecZm2xDTEkfPGRP1Bbr8U6n/view?usp=drive_link)

Before run this task, set up the [SemanticKITTI](http://www.semantic-kitti.org/) dataset first for semantic labels as input.

To run sampling on pretrained models (and to evaluate your results with flag "--eval"):
```
CUDA_VISIBLE_DEVICES=0 python scripts/sample_cond.py -r models/lidm/kitti/sem2lidar/model.ckpt -d kitti [--eval]
```

## Camera-to-LiDAR

<p align="center">
<img src=assets/cam2lidar.jpeg width="768"/>
</p>

Before run this task, set up the [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) dataset first for camera images as input.

To run sampling on pretrained models:
```
CUDA_VISIBLE_DEVICES=0 python scripts/sample_cond.py -r models/lidm/kitti/sem2lidar/model.ckpt -d kitti [--eval]
```


## Text-to-LiDAR

<p align="center">
<img src=assets/text2lidar.jpeg width="768"/>
</p>


To run sampling on pretrained models:
```
CUDA_VISIBLE_DEVICES=0 python scripts/text2lidar.py -r models/lidm/kitti/cam2lidar/model.ckpt -d kitti -p "an empty road with no object"
```

## Layout-to-LiDAR

Coming Soon...


## Acknowledgement

- Our codebase for the diffusion models builds heavily on [Latent Diffusion](https://github.com/CompVis/latent-diffusion)


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