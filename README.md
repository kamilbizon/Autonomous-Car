![My image](https://github.com/kamilbizon/Content/blob/master/auto.png)

# Autonomous-Car
#### The goal of our project was to learn how to use popular frameworks frequently used for research and other uses such as keras, numpy, OpenCV etc. The following script's goal is to clone behavior of a human driver for which we train a CNN model. For training and testing models we chose to use Udacity's Self Driving Car Simulator for the ease of use it provides and being faily lightweight. This allowed us to focus on other things in the project. That being said similar approach could be used both in real life or in video games such as GTA V.


## Getting started
#### The simplest way to install all dependencies is to install 64bit Anaconda with Pyhthon 3.6 version. Then run Anaconda Prompt and install all dependencies with:  
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
- start Udacity program in Training Mode, click Record and choose folder where you want to save data (recommend to make new folder in the same place where is training script). Click record once again and drive 3/4 laps.
- run training script, enter the name of folder with driving_log.csv file (you can choose training option i.e. used model):
```
python train.py -d [diretory]
```
- choose Autonomous Mode, trained model and run driving script, (you can try our model-003.h5, unfortunately it can't pass whole round):
```
python auto.py model-00x.h5
```
- if you used model_1 for training use '-1 True' option:
```
python auto.py model-00x.h5 -1 True
```
![My image](https://github.com/kamilbizon/Content/blob/master/drive.png)

## Training
#### For training we split data for training and validation part. First we preprocess and augment images to produce more data and prepare CNN for more possible situations:
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
## Model
#### Main model that we use is inpired by NVIDIA's model from project "End to End Learning for Self-Driving Cars"
![My image](https://github.com/kamilbizon/Content/blob/master/NVIDIA_model.png)
