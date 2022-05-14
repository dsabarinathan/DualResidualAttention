# DualResidualAttention
Dual  Residual Channel Attention Net for Spectral Reconstruction

This model has been created in the NTIRE 2022 Challenge on Spectral Reconstruction from RGB competition. We secured the 10th position in the competition. Our goal is to create a lighweight spectral reconstuction model. It can be used in mobile devices and normal computers. if we are able to get the high quality spectral images from RGB , we can find quality of the food prodcuts from mobile phone or normal laptop.

## Proposed network architecture diagram
![alt text](https://github.com/sabaridsn/DualResidualAttention/blob/main/architecture.png)

## Dual residual block diagram 
![alt text](https://github.com/sabaridsn/DualResidualAttention/blob/main/Dual_Residual_Channel_attentionDiagram.png)
## Environment

1. Python 3.6.4
2. Anaconda 4.9.2
3. Ubuntu 16.04 or Windows10

## How to setup the environment

#### Step 1 

Unzip the downloaded folder


#### Step 2

Open the powershell or terminal


#### Step 3

```
$cd yourpathtoLightWeightModel

$pwd
> ~/DualResidualAttention

$pip install --upgrade -r requirements.txt

```
## How to test the model on your own imgaes
```
$python test_v2.py --testImagePath=yourpathtoimages
```

## test results

| Data size  | Data  |  MRAE  |  RMSE  |
| :------: | :------: | :-------: | :-------: |  
| 100  | Training Data  | 0.73984  | 0.07509  |
| 50  | Validation Data  | 0.7795  | 0.1016  |

## Reference 

1. ["Coordinate 2D Convolution layer"](https://github.com/titu1994/keras-coordconv)
2. ["LightWeightModel"](https://github.com/sabaridsn/LightWeightModel)
