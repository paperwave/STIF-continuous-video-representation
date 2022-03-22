# STIF: Learning Continuous Video Representation for Space-Time Super-Resolution

## Prerequisites

- Python 3
- [PyTorch >= 1.1](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Deformable Convolution v2](https://arxiv.org/abs/1811.11168), we adopt [CharlesShang's implementation](https://github.com/CharlesShang/DCNv2) in the submodule.
- Python packages: `pip install numpy opencv-python lmdb pyyaml pickle5 matplotlib seaborn`

## Compile the DCNv2
```Shell
cd codes/models/modules/DCNv2
python setup.py install
```

## Test
```
cd codes
python custom_video_test.py
```
