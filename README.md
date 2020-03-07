# ABCNet

[Paper Link](https://arxiv.org/abs/2002.10200). Note this paper is not the final version. We will update soon.

We provide demo in this repository. We will update our code at [adet](https://github.com/aim-uofa/adet).

# Run Demo

Check [Installation](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md) for installation instructions.

```
bash vis_demo.sh
```

We assume that your symlinked `datasets/totaltext` directory has the following structure:

```
totaltext
|_ test_images
|  |_ 0000000.jpg
|  |_ ...
|  |_0000299.jpg
|_ annotations
|  |_ total_test.json 
```

Model [[Google Drive]](https://drive.google.com/open?id=1JiIhZXYE5VvT7f7BmaBbtDJThOkR36bo) 

Totaltext test data  [[Google Drive]](https://drive.google.com/open?id=1Y0fkBy0uy6uFKdlv6IVTZPvERqAoK_j2)

Syntext-150k [link] (Part1: 54,327 images. Part2: 94,723 images.)

# Description

<img src="demo/24.png" width="81%">

# Generated Results Download

Bezier-curve generated script [link](https://drive.google.com/open?id=1bFmdXCCsW0bj0qFgQl1MJarlWkwPSv_U).

[CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector) visualization results [[link]](https://drive.google.com/open?id=16PacExPoEspHS1k97-YJMFmDf-UEpPF6) (original training images can be downloaded [Here](https://github.com/Yuliang-Liu/Curve-Text-Detector/tree/master/data))

[Totaltext](https://github.com/cs-chan/Total-Text-Dataset) visualization results [[link]](https://drive.google.com/open?id=1LN-ZuVsajlAU-WAdJgTQsOwCSmGPzTPi)

# Cite

```BibTeX
@article{liu2020abcnet,
  title   =  {ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network},
  author  =  {Liu, Yuliang* and Chen, Hao* and Shen, Chunhua and He, Tong and Jin, Lianwen and Wang, Liangwei},
  journal =  {arXiv preprint arXiv:2002.10200},
  year    =  {2020}
}

```
* *represents the authors contributed equally.

# Copyright

Any suggestion is welcome. Please send email to liu.yuliang@mail.scut.edu.cn or yuliang.liu@adelaide.edu.au

For commercial purpose usage, please contact Dr. Lianwen Jin: eelwjin@scut.edu.cn

Copyright 2019, Deep Learning and Vision Computing Lab, South China China University of Technology. http://www.dlvc-lab.net
