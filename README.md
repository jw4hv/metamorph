# MetaMorph: Learning Metamorphic Image Transformation With Appearance Changes

This is a predictive model for both image segmentations & image registration for images with appearance changes, such as those caused by brain stroke and tumors. Its potential applications extend to real-time image-guided navigation systems, particularly in tumor removal surgery and related fields. Our paper is here, https://arxiv.org/abs/2303.04849

```Upcoming Additions```

We updated the most recent version of joint learning of MetaMorph. Note that we provide the training scripts, a testing procedure with a pretrained model and more advanced segmentation models, such as transformer-based techniques, will be inlcuded soon! 

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Parameters](#parameters)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)

## Introduction

Here is a overview of our entire model. Image segmentation: i). We first input a pair of images into a segmentation network, and apply predicted labels onto images to mask out the appearance change. Image registration: ii). We then input a pair of images (with masked-out appearance change) to the registration network and predict a piecewise velocity field, integrate geodesic constraints, and produce a deformed image and transformation-propagated segmentation. The deformed images and labels are circulated into the segmentation network as augmented data (more advanced augmentation strategies will be served as optional choices). We adopt jointly learning for both models. 

## Installation

The Python version needs be greater than 3.12.  

## Parameters

- optimizer: Specifies the optimizer used for training the model (e.g., 'Adam').
- scheduler: Specifies the learning rate scheduler (e.g., 'CosAn' for Cosine Annealing).
- loss: Specifies the loss function used for training (e.g., 'L2' for Mean Squared Error, NCC for normalized cross correlation and RMI for mutual information).
- augmentation: Boolean indicating whether data augmentation is enabled.
- reduced_dim: Specifies the reduced dimensions.
- pretrain_epoch: Specifies the number of epochs for pretraining.
- lr: Learning rate for both models. (Note that two separated learning rates can be adopted for both models, we will update a version on this soon!)
- epochs: Total number of epochs for training.
- batch_size: Batch size for training.
- weight_decay: Weight decay parameter.
- pre_train: Number of epochs for pre-training.
- Euler_steps: Number of steps for Euler integration.
- Alpha: Alpha in shooting.
- Gamma: Gamma in shooting.
- Lpow: the power of laplacian operator in shooting.

  
## Usage
To run training, simply execute the script Train_MetaMorph.py.

## Documentation
- The region-based mutual information paper (https://papers.nips.cc/paper_files/paper/2019/hash/a67c8c9a961b4182688768dd9ba015fe-Abstract.html), Region mutual information loss for semantic segmentation. Advances in Neural Information Processing Systems, 2019.
- Different shooting models we have included are, Stationary Velocity Field in Voxelmorph (Paper: https://arxiv.org/abs/1809.05231, Code: https://github.com/voxelmorph/voxelmorph); Vector-Momenta Shooting Method in Lagomorph (Paper: https://openreview.net/pdf?id=Hkg0j9sA1V, Code: https://github.com/jacobhinkle/lagomorph); LDDMM by Fourier representations (https://bitbucket.org/FlashC/flashc/src).

## Contributing
We are seeking collaboration opportunities to explore uncertainty quantification along tumor boundaries and related areas. If interested, please reach out to contribute or discuss further. 





