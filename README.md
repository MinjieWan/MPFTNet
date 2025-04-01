# MPFTNet
We provide the dataset of our work. You can download it by the following link: [dataset](https://pan.baidu.com/s/12EC2qY9ZBTGvlB386uLFtA) code:mkdn

# Installation

This repository is built in PyTorch 1.8.1 and tested on Ubuntu 16.04 environment (Python3.7, CUDA10.2, cuDNN7.6).
Follow these intructions

1. Make conda environment
```
conda create -n pytorch1.8 python=3.7
conda activate pytorch1.8
```

2. Install dependencies
```
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

3. Install basicsr
```
python setup.py develop --no_cuda_ext
```

# Training
```
cd MPFTNet
./train.sh Real_Denoising/Options/RealDenoising_MPFTNet.yml
```

# Testing
```
cd Real_Denoising
python test.py
```
