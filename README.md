# CoDis  
ICCV‘23: Combating Noisy Labels with Sample Selection
by Mining High-Discrepancy Examples (PyTorch implementation).  

This is the code for the paper:
[Combating Noisy Labels with Sample Selection
by Mining High-Discrepancy Examples](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_Combating_Noisy_Labels_with_Sample_Selection_by_Mining_High-Discrepancy_Examples_ICCV_2023_paper.pdf)      
Authors: Xiaobo Xia, Bo Han, Yibing Zhan, Jun Yu, Mingming Gong, Chen Gong, Tongliang Liu.

## Abstract
The sample selection approach is popular in learning with noisy labels. The state-of-the-art methods train two deep networks simultaneously for sample selection, which aims to employ their different learning abilities. To prevent two networks from converging to a consensus, their divergence should be maintained. Prior work presents that the divergence can be kept by locating the disagreement data on which the prediction labels of the two networks are different. However, this procedure is sample-inefficient for generalization, which means that only a few clean examples can be utilized in training. In this paper, to address the issue, we propose a simple yet effective method called CoDis. In particular, we select possibly clean data that simultaneously have high-discrepancy prediction probabilities between two networks. As selected data have high discrepancies in probabilities, the divergence of two networks can be maintained by training on such data. In addition, the condition of high discrepancies is milder than disagreement, which allows more data to be considered for training, and makes our method more sample-efficient. Moreover, we show that the proposed method enables to mine hard clean examples to help generalization. Empirical results show that CoDis is superior to multiple baselines in the robustness of trained models.


## Dependencies
we implement our methods by PyTorch. The environment is as bellow:
- [Ubuntu 16.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version >= 0.4.1
- [CUDA](https://developer.nvidia.com/cuda-downloads), version >= 9.0
- [Anaconda3](https://www.anaconda.com/)

Install PyTorch and Torchvision (Conda):
```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

Install PyTorch and Torchvision (Pip3):
```bash
pip3 install torch torchvision
```
## Experiments
We verify the effectiveness of CoDis on multiple datasets. We provide [datasets](https://drive.google.com/open?id=1Tz3W3JVYv2nu-mdM6x33KSnRIY1B7ygQ) (the images and labels have been processed to .npy format).        
Here is an example: 
```bash
python main.py --dataset mnist --noise_type symmetric --noise_rate 0.2
```

If you find this code useful in your research, please cite  
```bash
@inproceedings{xia2023combating,
  title={Combating Noisy Labels with Sample Selection by Mining High-Discrepancy Examples},
  author={Xia, Xiaobo and Han, Bo and Zhan, Yibing and Yu, Jun and Gong, Mingming and Gong, Chen and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1833--1843},
  year={2023}
}  
```
