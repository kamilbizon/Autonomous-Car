![My image](https://github.com/kamilbizon/Content/blob/master/auto.png)

# Autonomous-Car
#### Our project is designed to teach convolutional neural network (CNN) steer a car by cloning human drivers behaviour. We use program provided by Udacity which allow to get images and steering data (human drive) in tranining mode and let CNN steer a car in autonomous mode.


## Getting started
#### The simplest way to install all dependencies is install 64bit Anaconda with Pyhthon 3.6 version. Then run Anaconda Prompt and install all dependencies with:  
```
conda env create -f car.yml
```
#### Next activate environment:
```
conda activate car
```
#### Installing everything manually is not recommended.
#### Then you have to download precompiled Udacity program from https://github.com/udacity/self-driving-car-sim
#### To start with project:
- start Udacity program in Training Mode, click Record and choose folder where you want to save data (recommend to make new folder in the same place where is training script). Click record once again and drive 3/4 rounds.
- run training script, enter the name of folder with driving_log.csv file (you can choose training option i.e. used model):
```
python train.py -d [diretory]
```
- choose Autonomous Mode, trained model and run driving script:
```
python auto.py model-00x.h5
```

![My image](https://github.com/kamilbizon/Content/blob/master/drive.png)

## Training



#### For training we split data for training and validation part. First we preprocess images and do augmentation to produce more data and prepare CNN for more possible situations:
- reverse image and negate steering,
- change brigtness,
- add random shadow,
- translate image and steering,
- crop car and part of sky,
- translate rgb to yuv.

![My image](https://github.com/kamilbizon/Content/blob/master/shadow.png)
![My image](https://github.com/kamilbizon/Content/blob/master/brightness1.png)
![My image](https://github.com/kamilbizon/Content/blob/master/translate.png)
![My image](https://github.com/kamilbizon/Content/blob/master/yuv.png)

