# [Pytorch] Super-Resolution using Reinforcement Learning

Implementation of Super-Resolution model using Reinforement learning based on **Multi-Step Reinforcement Learning for Single Image Super-Resolution** paper with Pytorch.

## Contents
- [Introduction](#introduction)
- [Train](#train)
- [Test](#test)
- [Demo](#demo)
- [Evaluate](#evaluate)
- [References](#references)

## **Introduction**
This implementation using [PixelRL](https://arxiv.org/abs/1912.07190) as the core of Reinforement learning model, I use my 4 available Super-Resolution models for actions instead of 
EDSR and ESRGAN in this implementation. Because changing the actions are easy so you can use other super-resolution models as your actions.

<div align="center">

|Action index|     Action       |
|:----------:|:----------------:|
|     0      |  pixel value -= 1|
|     1      |  do nothing      |
|     2      |  pixel value += 1|
|     3      |      [ESPCN](https://github.com/Nhat-Thanh/ESPCN-Pytorch)       |
|     4      |      [SRCNN](https://github.com/Nhat-Thanh/SRCNN-Pytorch)       |
|     5      |      [VDSR](https://github.com/Nhat-Thanh/VDSR-Pytorch)         |
|     6      |      [FSRCNN](https://github.com/Nhat-Thanh/FSRCNN-Pytorch)     |
    
  <b>The actions table.</b>

</div>

## Train
Dataset:
   - Train: T91 + General100 + BSD100
   - Validation: Set14
   - Test: Set5

You run this command to begin the training:
```
python train.py --scale=2              \
                --steps=2000           \
                --batch-size=64        \
                --save-every=50        \
                --save-log=0           \
                --ckpt-dir="checkpoint/x2"
```
- **--save-log**: if it's equal to **1**, **train loss, train rewards, train metrics, validation rewards, validation metrics** will be saved every **save-every** steps.

**NOTE**: if you want to re-train a new model, you should delete all files in **checkpoint** directory. Your checkpoint will be saved when above command finishs and can be used for the next times, so you can train a model on Google Colab without taking care of GPU time limit.

I trained the model on Google Colab in 2000 steps:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nhat-Thanh/PixelRL-SR/blob/main/PixelRL-SR.ipynb)

You can get the weights of models here: 
- [PixelRL_SR-x2.pt](checkpoint/x2/PixelRL_SR-x2.pt)
- [PixelRL_SR-x3.pt](checkpoint/x3/PixelRL_SR-x3.pt)
- [PixelRL_SR-x4.pt](checkpoint/x4/PixelRL_SR-x4.pt)

The log informations of my training process are plot below, these plot line are smoothed by [Exponential Moving Average](https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average) with alpha = 0.2:
<div align="center">
  <br>
  <img src="README/scale-x2.png" width=800/></br>
  <b>Loss (Left), reward (center), PSNR (right) of scale x2.</b>
  
  <br>
  <br>
  <img src="README/scale-x3.png" width=800/></br>
  <b>Loss (Left), reward (center), PSNR (right) of scale x3.</b>
    
  <br>
  <br>
  <img src="README/scale-x4.png" width=800/></br>
  <b>Loss (Left), reward (center), PSNR (right) of scale x4.</b>
</div>


## Test
I use **Set5** as the test set. After Training, you can test models with scale factors **x2, x3, x4**, the result is calculated by compute average PSNR of all images.
```
python test.py --scale=2 --ckpt-path="default"
```

**--ckpt-path="default"** mean you are using default weights path, aka **checkpoint/x{scale}/PixelRL_SR-x{scale}.pt**. If you want to use your trained weights, you can pass yours to **--ckpt-path**.

## Demo 
After Training, you can test models with this command, the result is the **sr.png**.
```
python demo.py --scale=2             \
               --ckpt-path="default" \
               --draw-action-map=0   \
               --image-path="dataset/test2.png"
```

**--draw-action-map**: If it's equal to **1**, an action map will be save to **action_maps** directory every step.

**--ckpt-path** is the same as in [Test](#test)

## Evaluate

I evaluated models with Set5, Set14, BSD100 and Urban100 dataset by PSNR. I use Set5's Butterfly to show my result:

<div align="center">

|  Dataset  |   Set5  |  Set14  |  BSD100 | Urban100 |
|:---------:|:-------:|:-------:|:-------:|:--------:|
|     x2    | 38.7333 | 34.4900 | 34.4501 | 31.6963  |
|     x3    | 34.6700 | 31.3102 | 31.3425 |     X    |
|     x4    | 32.0406 | 29.3773 | 29.7230 | 27.0564  |

  <br/>

  <img src="./README/example.png" width="1000"/><br/>
  <b>Bicubic (left), PixelRL-SR x2 (center), High Resolution (right).</b>
</div>

## References
- Multi-Step Reinforcement Learning for Single Image Super-Resolution: https://ieeexplore.ieee.org/document/9150927
- Fully Convolutional Network with Reinforcement Learning for Image Processing: https://arxiv.org/abs/1912.07190
- T91, General100, BSD200: http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_training_datasets.zip
- Set5: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Set5_SR.zip
- Set14: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Set14_SR.zip
- BSD100: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/BSD100_SR.zip
- Urban100: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Urban100_SR.zip
