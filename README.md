<div align="center">
<h1>LiDAR Diffusion Models [CVPR 2024]</h1>

[**Haoxi Ran**](https://hancyran.github.io/) · [**Vitor Guizilini**](https://scholar.google.com.br/citations?user=UH9tP6QAAAAJ&hl=en) · [**Yue Wang**](https://yuewang.xyz/)


<a href="https://hancyran.github.io/assets/paper/lidar_diffusion.pdf"><img src='https://img.shields.io/badge/PDF-LiDAR Diffusion-yellow' alt='PDF'></a>
<a href="https://arxiv.org/abs/2404.00815"><img src='https://img.shields.io/badge/arXiv-2404.00815-red?logo=arXiv' alt='arXiv'></a>
<a href="https://lidar-diffusion.github.io/"><img src='https://img.shields.io/badge/Project-LiDAR Diffusion-green' alt='Project'></a>
<a href="https://www.youtube.com/watch?v=Vj7DubNZnDo"><img src='https://img.shields.io/badge/youtube-Video-slateblue?logo=youtube' alt='Video'></a>
<a href="#citation"><img src='https://img.shields.io/badge/BibTex-LiDAR Diffusion-blue' alt='Paper BibTex'></a>


<img src=assets/overview.png width="400"/>

</div>

## :tada: News :tada:

- [**Apr 14, 2024**] Pretrained autoencoders and LiDMs for different tasks are released!
- [**Apr 5, 2024**] Our codebase and a detailed study of our autoencoder design along with the pretrained models is released! 



## Requirements

We provide an available [conda](https://conda.io/) environment named `lidar_diffusion`:

```
sh init/create_env.sh
conda activate lidar_diffusion
```

## Evaluation Toolbox

**Overview of evaluation metrics**:

<table>
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

#### 64-beam (evaluated on KITTI-360 val):

| Encoder  | rFRID(↓) | rFSVD(↓) | rFPVD(↓) | CD(↓) | EMD(↓) |                                                        Checkpoint                                                        |                        Rec.&nbsp;Results&nbsp;val<br/>(Point&nbsp;Cloud)                         |         Comment          |
|:--------:|:--------:|:--------:|:--------:|:-----:|:------:|:------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|:------------------------:|
| f_c2_p4  |   2.15   |   20.2   |   16.2   | 0.160 | 0.203  | [[Google&nbsp;Drive]](https://drive.google.com/file/d/1fUlQVqnShylps4-PnFCRD-sW-6v_rAB4/view?usp=drive_link)<br/>(205MB) | [[Video]](https://drive.google.com/file/d/1bIjRtrF3ljtcR-esjTL79uJisn4cNf2D/view?usp=drive_link) |                          |
| f_c2_p4* |   2.06   |   20.3   |   15.7   | 0.092 | 0.176  | [[Google&nbsp;Drive]](https://drive.google.com/file/d/1A0zhQQXZTr8IfvpmsXrsG3lISC8KLkka/view?usp=drive_link)<br/>(205MB) | [[Video]](https://drive.google.com/file/d/1P_FbIOmYtS3kgutVAYXr7RShryO5Md7s/view?usp=drive_link) | *: w/o logarithm scaling |


### Benchmark for Unconditional LiDAR Generation

#### 64-beam (2k samples):

|         Method         | Encoder  |  FRID(↓)  | FSVD(↓)  | FPVD(↓)  |  JSD(↓)   | MMD<br/>(10^-4,↓) |                                                        Checkpoint                                                        |                                Output&nbsp;LiDAR<br/>Point&nbsp;Clouds                                |
|:----------------------:|:--------:|:---------:|:--------:|:--------:|:---------:|:-----------------:|:------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------:|
|       LiDAR-GAN        |          |   1222    |  183.4   |  168.1   |   0.272   |       4.74        |                                                            -                                                             | [[2k samples]](https://drive.google.com/file/d/1lzOqXHxtO83HNMZ_7_dU9GMee_Zm3clO/view?usp=drive_link) |
|       LiDAR-VAE        |          |   199.1   |  129.9   |  105.8   |   0.237   |       7.07        |                                                            -                                                             | [[2k samples]](https://drive.google.com/file/d/1_6KGATYfzLur9bt8vISLXwzEIsjbXq_k/view?usp=drive_link) |
|      ProjectedGAN      |          |   149.7   |   44.7   |   33.4   |   0.188   |       2.88        |                                                            -                                                             | [[2k samples]](https://drive.google.com/file/d/1LzLhuKpBOIZ6F7SlPtMdYuCSE_8P1qwz/view?usp=drive_link) |
|      UltraLiDAR§       |          |   370.0   |   72.1   |   66.6   |   0.747   |       17.12       |                                                            -                                                             | [[2k samples]](https://drive.google.com/file/d/17kft5S0nA_lnjECrK_aHzI5q1Erma_T7/view?usp=drive_link) |
| LiDARGen&nbsp;(1160s)† |          |   129.0   |   39.2   |   33.4   | **0.188** |     **2.88**      |                                                            -                                                             | [[2k samples]](https://drive.google.com/file/d/1N5jTHjM8XnUYAMYkbsOipUQGhZjqMYDD/view?usp=drive_link) |
|                        |          |           |          |          |           |                   |                                                                                                                          |                                                                                                       |
|  LiDARGen&nbsp;(50s)†  |          |   2051    |  480.6   |  400.7   |   0.506   |       9.91        |                                                            -                                                             | [[2k samples]](https://drive.google.com/file/d/1qN4T0Jg8P4IJLdaR_7sBdjID3TtzLITy/view?usp=drive_link) |
|    LiDM&nbsp;(50s)     | f_c2_p4  |   135.8   | **37.9** | **28.7** |   0.211   |       3.87        | [[Google&nbsp;Drive]](https://drive.google.com/file/d/1WKFwXi7xiXr2WCtM3ZX95CqlU-kOhhgC/view?usp=drive_link)<br/>(3.9GB) | [[2k samples]](https://drive.google.com/file/d/1mdWdzXHTW4IONgAYD44EvfUI8aokPfP_/view?usp=drive_link) |
|    LiDM&nbsp;(50s)     | f_c2_p4* | **125.1** |   38.8   |   29.0   |   0.211   |       3.84        | [[Google&nbsp;Drive]](https://drive.google.com/file/d/1huCr1xQJ6ZRS2VYcJ99vDrCS8QhxVysQ/view?usp=drive_link)<br/>(3.9GB) | [[2k samples]](https://drive.google.com/file/d/18K-9ps9Ej-OACRKe7D30reY4l6CttN6T/view?usp=drive_link) |

NOTE:
1. Each method is evaluated with **2,000** randomly generated samples. 
2. †: samples generated by the officially released pretrained model in [LiDARGen github repo](https://github.com/vzyrianov/lidargen).
3. §: samples borrowed from [UltraLiDAR implementation](https://github.com/myc634/UltraLiDAR_nusc_waymo).
4. All above results are calculated from our [evaluation toolbox](#evaluation-toolbox). For more details, please refer to [Evaluation Toolbox README](./lidm/eval/README.md).
5. Each .pcd file is a list of point clouds stored by `joblib` package. To load those files, use command `joblib.load(path)`.

To evaluate above methods (except _LiDM_) yourself, download our provided .pcd files in the **Output** column to directory `./models/baseline/kitti/[method]/`:

```
CUDA_VISIBLE_DEVICES=0 python scripts/sample.py -d kitti -f models/baseline/kitti/[method]/samples.pcd --baseline --eval
```

To evaluate LiDM through the given .pcd files:

```
CUDA_VISIBLE_DEVICES=0 python scripts/sample.py -d kitti -f models/lidm/kitti/[method]/samples.pcd --eval
```

### Pretrained LiDMs for Other Tasks

|                 Task                 | Encoder  |                         Dataset                         | FRID(↓) | FSVD(↓) |                                                        Checkpoint                                                        |                                                      Output                                                       |
|:------------------------------------:|:--------:|:-------------------------------------------------------:|:-------:|:-------:|:------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|
| Semantic&nbsp;Map&nbsp;to&nbsp;LiDAR | f_c2_p4* | [SemanticKITTI](http://semantic-kitti.org/dataset.html) |  11.8   |  19.1   | [[Google&nbsp;Drive]](https://drive.google.com/file/d/1Mijx3cRPupsC2d4b2FwlbOXsojeHAXaO/view?usp=drive_link)<br/>(3.9GB) | [[log.tar.gz]](https://drive.google.com/file/d/1N2hMDO0boL5TPmnulApPspIpNnG5d9e5/view?usp=drive_link)<br/>(2.1GB) |
|      Camera&nbsp;to&nbsp;LiDAR       | f_c2_p4* | [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) |  38.9   |  32.1   | [[Google&nbsp;Drive]](https://drive.google.com/file/d/1XzY7fSHQz72gWVFcmit-NlkoSwtMWbfz/view?usp=drive_link)<br/>(7.5GB) | [[log.tar.gz]](https://drive.google.com/file/d/1PZrMwiZiVvpYuuMKxMHpWEalt0b1lM17/view?usp=drive_link)<br/>(5.4GB) |
|       Text&nbsp;to&nbsp;LiDAR        | f_c2_p4* |                       _zero-shot_                       |    -    |    -    |                                               From&nbsp;_Camera-to-LiDAR_                                                |                                                         -                                                         |

NOTE:
1. The output `log.tar.gz` contains input conditions (`.png`), generated range images (`.png`), generated point clouds (`.txt`), and a collection of all output point clouds (`.pcd`). 


### Study on Design of LiDAR Compression 

For full details of our studies on the design of LiDAR Compression, please refer to [LiDAR Compression Design README](./DESIGN.md).

Tip: Download the video instead of watching it with the Google Drive's built-in video player provides a better visualization.

#### Autoencoders (trained with 40k steps, evaluated on reconstruction):

| Curvewise <br/> Factor | Patchwise <br/> Factor | Output <br/> Size | rFRID(↓) | rFSVD(↓) | #Params (M) |                                                                                          Visualization of Reconstruction (val)                                                                                          |
|:----------------------:|:----------------------:|:-----------------:|:--------:|:--------:|:-----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|          N/A           |          N/A           |   Ground Truth    |    -     |    -     |      -      | [[Range&nbsp;Image]](https://drive.google.com/file/d/1wAtQSlVwF2jCpcL3zbXlk2lGUYzo1GBf/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/1iHIB7Jw-WS0D_hXgQSOyyDyWCmPVR-6k/view?usp=sharing) |
|                        |                        |                   |          |          |             |                                                                                                                                                                                                                         |
|           4            |           1            |     64x256x2      |   0.2    |   12.9   |    9.52     | [[Range&nbsp;Image]](https://drive.google.com/file/d/1w7slbsRjlU4kb0kl6LyjX-JojJvoWQhG/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/17ewPXoRMeA_HsvEOznsvxy3d6iKk7hC2/view?usp=sharing) |
|           8            |           1            |     64x128x3      |   0.9    |   21.2   |    10.76    | [[Range&nbsp;Image]](https://drive.google.com/file/d/17kukYFlJY40_cVBuWXMLHiMe7ls2OLNh/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/116IXDMgrWn6OHtyEYIo6aM1ARloX3BWF/view?usp=sharing) |
|           16           |           1            |      64x64x4      |   2.8    |   31.1   |    12.43    | [[Range&nbsp;Image]](https://drive.google.com/file/d/12TKyoajTiU_hr1MAdK2PNveddorCshG4/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/18NCV7JoR3W1COaPH96a1ozbh8-58eT6n/view?usp=sharing) |
|           32           |           1            |      64x32x8      |   16.4   |   49.0   |    13.72    | [[Range&nbsp;Image]](https://drive.google.com/file/d/1S2DPHfWAljKZrHJlPHIvxAPK2-rpdJ_J/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/1yx8V4Qav7sCigcfSHrrrJQOFF-s2PryV/view?usp=sharing) |
|                        |                        |                   |          |          |             |                                                                                                                                                                                                                         |
|           1            |           2            |     32x512x2      |   1.5    |   25.0   |    2.87     | [[Range&nbsp;Image]](https://drive.google.com/file/d/1tPPD2Pnn_6ge3x2yoJXhkDhe0Wi5Qxhw/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/1Xjg0ckVb208BFEgbv4VQtV-fVraEXUNC/view?usp=sharing) |
|           1            |           4            |     16x256x4      |   0.6    |   15.4   |    12.45    | [[Range&nbsp;Image]](https://drive.google.com/file/d/1Q_ZTRKyDOAmP314p9B6Cip79mc-FJ2se/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/1-t9zvSrov1OsF_WEIBqH3xkLzTJfxRBr/view?usp=sharing) |
|           1            |           8            |     8x128x16      |   17.7   |   35.7   |    15.78    | [[Range&nbsp;Image]](https://drive.google.com/file/d/14hPy2utsaxwPxW5PA7gO7ak7f-lcd-X5/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/1izj-_1hFkdaRCg2qUzkXByfCD-vBd_1M/view?usp=sharing) |
|           1            |           16           |      4x64x64      |   37.1   |   68.7   |    16.25    | [[Range&nbsp;Image]](https://drive.google.com/file/d/1G7evMm3H6WvbHFhBlCa8wxPzwVC3q-8H/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/1IdBrEpCIugvxVHyNOsNIg8Y8ZBWrHcWL/view?usp=sharing) |
|                        |                        |                   |          |          |             |                                                                                                                                                                                                                         |
|           2            |           2            |     32x256x3      |   0.4    |   11.2   |    13.09    | [[Range&nbsp;Image]](https://drive.google.com/file/d/1Ac4jVB6RkqMwV1fZcPGDyQhR3eE_Zj6C/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/1pg2ezSmXiu3ensvj564JIy6CpB46uZm7/view?usp=sharing) |
|           4            |           2            |     32x128x4      |   3.9    |   19.6   |    14.35    | [[Range&nbsp;Image]](https://drive.google.com/file/d/1yZGqe_DcDXew3JabnN4T1-P27ZlscHba/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/1i_q6gVY4gMtzKYlhlMQ9QrRql73VX05j/view?usp=sharing) |
|           8            |           2            |      32x64x8      |   8.0    |   25.3   |    16.06    | [[Range&nbsp;Image]](https://drive.google.com/file/d/1HVqFbIE1lgotDplc8x7_hJkSU5vLtbRN/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/1jSYWZMmPelmfWpVa7V5f2Byr9vN2BKXo/view?usp=sharing) | 
|           16           |           2            |     32x32x16      |   21.5   |   54.2   |    17.44    | [[Range&nbsp;Image]](https://drive.google.com/file/d/1flAzjRLcl5Jtc_T--GbbomKWi42DvW9v/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/1zfMzu6NFeLJhR1YPU28k7vPy1GX-80QT/view?usp=sharing) |
|           2            |           4            |     16x128x8      |   2.5    |   16.9   |    15.07    | [[Range&nbsp;Image]](https://drive.google.com/file/d/1rm0sviRg4LfImgWVCi6THi3pHF4kFccH/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/1gPKB2zj44oLLEBuUXU8uiaXIcSWpyMOi/view?usp=sharing) |
|           4            |           4            |     16x128x16     |   13.8   |   29.5   |    16.86    | [[Range&nbsp;Image]](https://drive.google.com/file/d/1ldMRXfUtFNBtjCCc-KYR311dQvCmn0EF/view?usp=sharing),&nbsp;[[Point&nbsp;Cloud]](https://drive.google.com/file/d/129WcZXW3b6e4UMxZ9x4XCR3BlaKw1Vec/view?usp=sharing) |


## Unconditional LiDAR Generation

<p align="center">
<img src=assets/uncond.jpeg width="512"/>
</p>

To run sampling on pretrained models (and to evaluate your results with flag "--eval"), firstly download our provided [pretrained autoencoders](#pretrained-autoencoders) to directory `./models/first_stage_models/kitti/[model_name]` and [pretrained LiDMs](#benchmark-for-unconditional-lidar-generation) to directory `./models/lidm/kitti/[model_name]`:

```
CUDA_VISIBLE_DEVICES=0 python scripts/sample.py -d kitti -r models/lidm/kitti/[model_name]/model.ckpt -n 2000 --eval
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

## Training


Besides, to train your own LiDAR Diffusion Models, just run this command (for example, train both autoencoder and lidm on four gpus):
```
# train an autoencoder
python main.py -b configs/autoencoder/kitti/autoencoder_c2_p4.yaml -t --gpus 0,1,2,3

# train an LiDM
python main.py -b configs/lidar_diffusion/kitti/uncond_c2_p4.yaml -t --gpus 0,1,2,3
```

To debug the training process, just add flag `-d`:
```
python main.py -b path/to/your/config.yaml -t --gpus 0, -d
```

To resume your training from an existing log directory or an existing checkpoint file, use the flag `-r`:
```
# using a log directory
python main.py -b path/to/your/config.yaml -t --gpus 0, -r path/to/your/log

# or, using a checkpoint 
python main.py -b path/to/your/config.yaml -t --gpus 0, -r path/to/your/ckpt/file
```


## Acknowledgement

- Our codebase for the diffusion models builds heavily on [Latent Diffusion](https://github.com/CompVis/latent-diffusion)


## Citation

If you find this project useful in your research, please consider citing:
```
@inproceedings{ran2024towards,
    title={Towards Realistic Scene Generation with LiDAR Diffusion Models},
    author={Ran, Haoxi and Guizilini, Vitor and Wang, Yue},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```
