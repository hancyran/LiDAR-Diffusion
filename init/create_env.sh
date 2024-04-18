#!/usr/bin/bash

# install rust compiler
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

# create conda environment
conda create -n lidar_diffusion python=3.10.11 -y
conda activate lidar_diffusion

# install dependencies
pip install --upgrade pip
pip install torchmetrics==0.5.0 pytorch-lightning==1.4.2 omegaconf==2.1.1 einops==0.3.0 transformers==4.36.2 imageio==2.9.0 imageio-ffmpeg==0.4.2 opencv-python kornia==0.7.0 wandb more_itertools
pip install gdown
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip

# install torchsparse (optional)
#apt-get install libsparsehash-dev
#pip install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

mkdir -p dataset/
