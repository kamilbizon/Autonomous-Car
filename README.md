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
- start Udacity program in training mode, click Record and choose folder where you want to save data (recommend to make new folder in the same place where is training script). Click record once again and drive 3/4 rounds.
- run training script:
```
python 

