# Efficient Light Field Image Super-Resolution via Progressive Disentangling

This repository contains official pytorch implementation of Efficient Light Field Image Super-Resolution via Progressive Disentangling, accepted by NTIRE 2024, done by Gaosheng Liu, Huanjing Yue, and Jingyu Yang.
![Network](https://github.com/GaoshengLiu/PDistgNet/blob/main/fig/PDistgNet.png)  

## Dataset
We use the processed data by [LF-DFnet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9286855), including EPFL, HCInew, HCIold, INRIA and STFgantry datasets for training and testing. Please download the dataset in the official repository of [LF-DFnet](https://github.com/YingqianWang/LF-DFnet).
## Code
### Dependencies
* Ubuntu 18.04
* Python 3.6
* Pyorch 1.3.1 + torchvision 0.4.2 + cuda 92
* Matlab
### Prepare Test Data
* Please refer to the previous work for data generation
### Test
* Run:
  ```python
  python test.py

## Citation
If you find this work helpful, please consider citing the following papers:<br> 
```Citation
@inproceedings{liu2024efficient,
  title={Efficient Light Field Image Super-Resolution via Progressive Disentangling},
  author={Liu, Gaosheng and Yue, Huanjing and Yang, Jingyu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={6277--6286},
  year={2024}
}
```


