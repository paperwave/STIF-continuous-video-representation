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

## Testing
Change the data path in [custom_video_test.py line 57](https://github.com/zychen-ustc/STIF-continuous-video-representation/blob/4f41fe924c5308b7529c853bc7b344b822d2e3a5/codes/custom_video_test.py#L57) and put video sequences in it. The file structure is as follows:

```
data_path
├── sequence1
    ├── im01.png
    ├── ...
    ├── im99.png
```

For testing:
```
cd codes
python custom_video_test.py
```
