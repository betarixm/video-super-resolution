Video Super Resolution
===
[![CSE@POSTECH](https://img.shields.io/badge/CSED451-POSTECH-c80150)](https://www.postech.ac.kr)

Team `한국인`
- 도승욱 `20180331` `컴퓨터공학과`
- 최은수 `20180050` `컴퓨터공학과`
- 권민재 `20190084` `컴퓨터공학과`


# Introduction
## Background
최근 4K에 이어 8K 기술까지 고화질, 고해상도의 display기술이 집약적으로 발전하면서 그에 맞추어 고화질 영상의 수요가 폭발적으로 증가하고 있다. 새로운 고해상도 영상을 새롭게 제작하는 것도 좋지만, 과거에 제작된 저해상도의 영화, 비디오 등의 영상 컨텐츠를 고해상도 device에서 세밀하고 촘촘한 pixel로 표현된 video로  시청하고자 하는 수요도 증가하고 있다. 이러한 문제를 해결하기 위하여 Video의 해상도를 올리는 Video upscaling기술도 발맞추어 발전하고 있다. 기존에는 Interpolation Algorithm을 사용하여 주변의 픽셀 값으로 모르는 데이터 값을 추정하는 방식을 사용했다. 하지만 이는 이미지의 크기만 조정해 주며 품질 자체는 개선시키지 못하기 때문에 Deep-learning을 사용한 upscaling 방법이 사용되고 있다. 그래서 우리는 이러한 시대적 수요에 맞추어 video upscaling을 텀프로젝트 주제로 선정하였다.

## Description
#### Goal
우리의 목표는 기존의 4k미만의 저화질 영상을 정수배 개선된 해상도를 가지는 고화질 영상으로 upscaling하는 프로그램을 제작하는 것이 목표이다.

#### Method
이 때 우리는 다양한 방법 중 deep-learning을 활용하고자 “Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit” 이라는 CVPR 논문의 방법을 차용할 것이다. 기존 논문의 구현에 가능하다면 개선점을 찾는 것까지 폭 넓게 목표하고 있다.

#### Data set
Deep-Learning에 사용되는 video data는 화질 별 선택 수집이 가능한 youtube에서 저작권 문제가 없는 video를 선택하여 크롤링한 후 가공하여 사용할 예정이다. 가공된 데이터는 ffmpeg 라는 라이브러리를 사용하여 활용 할 것이다.

#### Final Goal
최종적으로는 flask를 이용해 간단한 Web application을 제작하여 유저들이 웹사이트에 upscaling을 원하는 임의의 video를 업로드 하면 upscaling 후 download 받을 수 있도록 구현할 예정이다.


# Design
## Problem Analysis
 **Super Resolution**은 그래픽스 분야의 한 갈래로, 저해상도의 미디어를 고해상도의 미디어로 변환하는 것을 말한다. 이 목적 자체가 도전적이기 때문에 많은 사람들이 도전하고 있는 동시에, 과거 해상도가 낮게 기록된 미디어나 화질 면에서 손상된 미디어를 복구해낼 수 있다는 점에서 흥미로운 주제로 여겨진다. 기존에는 Bicubic Interpolation과 같은 수학적인 방법을 이용하여 픽셀 보간을 채우는 방식으로 Super Resolution의 목적을 달성하였으나, 최근에는 Deep Learning이 여러 분야에 도입됨에 따라 Super Resolution에서도 deep learning을 이용하는 추세이다. Super Resolution은 주로 이미지나 비디오에서 사용된다. 비디오는 연속된 이미지의 더미로 구성되며, 1초에 일정한 양의 이미지를 넘기는 방식으로 구현된다. 그렇기 때문에 비디오에 대한 Super Resoltion을 진행하는 것은 이미지에 비해 더 어렵지만, 최근 영상 기반 미디어 산업이 시시각각으로 발전함에 따라 사용처가 매우 늘어났기 때문에 이미지 해상도를 높이는 것 보다 더 활용성이 뛰어날 것으로 기대된다.

## Research Directions
Jo, Y., Oh, S. W., Kang, J., & Kim, S. J. (2018). Deep video super-resolution network using dynamic upsampling filters without explicit motion compensation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3224-3232).

### Problem Definition
Video super resolution의 목표는 주어진 low resolution(LR) frame $\{X_t\}$로부터 high resolution(HR) frame $\{\hat{Y_t}\}$을 획득하는 것이다. VU network $G$와 network parameter $\theta$로부터 다음과 같이 VU problem을 정의한다.
$$
    \hat{Y_t} = G_{\theta}(X_{t-N:t+N})
$$
이 때, $N$은 temporal radius를 의미한다. $G$의 input tensor shape는 $T * H * W * C$이며, $T= 2N + 1$, $H$와 $W$는 각각 LR frame의 height와 weight을 의미한다. 그리고 $C$는 color channel의 개수이다. output tensor shape는 $1 * rH * rW * C$이다. $r$은 upscaling factor이다. 

### Network Architecture
$\hat{Y_t}$를 생성하기 위해 network는 $\{X_{t-N:t+N}\}$로부터 dynamic upscaling filter F_t와 residual R_t를 생성한다. 그 후, Input center frame X_t는 먼저 F_t로 upscaling 된 후, R_t와 더해져 최종적으로 \hat{Y_t}가 생성된다.
![](https://i.imgur.com/KQ8VyqI.png)


#### Dynamic Upsampling Filters
훈련된 network는 $\{X_{t-N:t+N}\}$을 input으로 받아 특정 사이즈의 filter들의 set $F_t$를 output한다. ($F_t$는 $r^2HW$개의 filter로 구성) 그리고 이것은 filtered HR frame $\tilde{Y_t}$를 생성하는데 사용된다. 각각의 HR pixel value는 input frame $X_t$의 LR pixel에 local filtering을 filter $F_t^{y,x,v,u}$로 적용함으로써 얻을 수 있다.
$
\tilde{Y_t}(yr+v, xr + u) = \sum_{j=-2}^2\sum_{i=-2}^2F_t^{y,x,v,u}(j+2, i+2)X_t(y+j, x+i)
$
여기서, $x$와 $y$는 LR grid의 좌표이며, $v$와 $u$는 $r*r$ output block의 corrdinate이다.
![](https://i.imgur.com/edFKPUS.png)


#### Residual Learning
dynamic upsampling filter만을 적용시킨 결과는 sharpness에서 lack을 가진다. 그건 그저 input pixel의 weighted sum이기 때문이다. 이것을 해결하기 위해 residual image를 estimate하여 high frequency detail을 증가시킨다. Dynamic Upsampling process와 residual process를 결합함으로써 HR frame에서 spatial sharpness와 temporal consistency를 성취할 수 있다.

### Temporal Augmentation
dynamic upsampling filter만을 적용시킨 결과는 sharpness에서 lack을 가진다. 그건 그저 input pixel의 weighted sum이기 때문이다. 이것을 해결하기 위해 residual image를 estimate하여 high frequency detail을 증가시킨다. Dynamic Upsampling process와 residual process를 결합함으로써 HR frame에서 spatial sharpness와 temporal consistency를 성취할 수 있다.
![](https://i.imgur.com/DVvy3X8.png)


# Implementation
## Dataset 
### crawling
저작권 문제가 없는 동영상 수집을 위해 “pixabay”에서 “상업적 용도로 사용 가능 / 출처 안 밝혀도 됨” 표시가 되어 있는 video 중에서  motion이 dynamic한 영상들로 수집하였다. 
Pixabay python module을 사용하여 자동 crawling을 했으며, "Walking", "Animal", "Exercise", "Transport" 등의 keyword로 dynamic한 motion이 포함되어 있고, 길이가 8~15sec 이내의 video 들로만 crawling 하였다.

### convert to Frame
opencv library를 사용하여 총 670개의 video,  대략 320,000 개의 frame image를 추출하였습니다. Ground truth는 frame image 중 모션이 가장 잘 나타난다고 추정할 수 있는 중앙 부분의 128x128 크롭된 image를 사용하였고, LR input은 32x32 frame image를 사용하였습니다.

## Network architecture

## Training
제작한 dataset과 Network를 사용하여 Training 을 진행하였다. 컴공과 클러스터 서버의 TitanXP GPU를 사용하였고, job 1개에 100 Epoch, Batch size 16, GPU2개로 학습하였다. 클러스터 서버의 최대 job processing 시간이 24시간으로 제한되어 있어서 100Epoch로 제한하고 여러번 train하였다.
학습에 사용한 hyperparameter는 다음과 같으며 조금씩 변경하며 validation loss를 줄이는 방향으로 학습을 진행하였다. 
```
    (1) Initial LR     
    (2) First decay step
    (3) Training set    
    (4) Huber delta    
```
    

## Web Application






# Result
## Hyperparameter
최종적으로 validation loss = 0.00435, hyperparameter는 다음과 같다.
```
INIT_LR = 0.005
FIRST_DECAY_STEPS = 70
TRAIN_DATASET_RATIO = 0.9
HUBER_DELTA= 200.0
```

## upscaling 



# Discussion
[각자 하나씩은 쓰자]
### Motion Area identification
논문에서는 dataset을 제작할 때, video frame을 그대로 사용하지 않고 "motion이 충분히 보이는 area" 를 crop하여 일부분만 사용하였다. 비슷하게 motion이 충분한 area를 crop하여 dataset을 제작하고자 하였으나 300개가 넘는 video를 일일이 확인하여 crop 위치를 결정하는 것은 무리가 있었다. 그래서 대체적으로 frame의 중심(가운데) 에 Object의 motion이 활발하게 보인다고 가정하고 중심부분을 crop하여 제작하였다. 그렇게 하다 보니 video에 따라서는 motion이 충분하지 않아 training에 도움이 되지 못하는 data들도 있었다. 논문에서는 300개의 video로 충분한 결과를 도출하였지만, 본 project에서는 dataset이 더 필요하게 되었고 추가적으로 300개의 video를 더 crawling해서 최종 dataset을 제작하였다. 이러한 Motion area를 identify할 수 있는 방법에 대해 좀 더 고민해 본다면 적은 dataset으로도 더 좋은 성능을 낼 수 있을 것이다. 






