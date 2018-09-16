from image_processing import INPUT_SHAPE, second_shape
from image_processing import batch_NVIDIA_gen, batch_gen
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten


class model(ABC):
    @abstractmethod
    def train(self, args, x_train, x_valid, y_train, y_valid):
        pass

    @abstractmethod
    def build(self, args):
        pass


class NVIDIA_model(model):
    def train(self, args, x_train, x_valid, y_train, y_valid):
        self.checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                     monitor = 'val_loss',
                                     verbose = 0,
                                     save_best_only = args.keep_best,
                                     mode = 'auto')

        self.model.compile(loss = 'mean_squared_error', optimizer = Adam(lr = args.learn_rate))

        self.model.fit_generator(batch_NVIDIA_gen(args.data_dir, x_train, y_train, args.batch_size, True),
                                 args.samples,
                                 args.num_epochs,
                                 max_q_size = 1,
                                 validation_data = batch_NVIDIA_gen(args.data_dir, x_valid, y_valid, args.batch_size, False),
                                 nb_val_samples = len(x_valid),
                                 callbacks = [self.checkpoint],
                                 verbose = 1)
        
    def build(self, args):
        self.model = Sequential()
        self.model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape = INPUT_SHAPE))
        self.model.add(Conv2D(24, 5, 5, activation = 'elu', subsample = (2, 2)))
        self.model.add(Conv2D(36, 5, 5, activation = 'elu', subsample = (2, 2)))
        self.model.add(Conv2D(48, 5, 5, activation = 'elu', subsample = (2, 2)))
        self.model.add(Conv2D(64, 3, 3, activation = 'elu'))
        self.model.add(Conv2D(64, 3, 3, activation = 'elu'))
        self.model.add(Dropout(args.keep_prob))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation = 'elu'))
        self.model.add(Dense(50, activation = 'elu'))
        self.model.add(Dense(10, activation = 'elu'))
        self.model.add(Dense(1))
        self.model.summary()


class model_1(model):
    def train(self, args, x_train, x_test, y_train, y_test):
        self.checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                     monitor = 'val_loss',
                                     verbose = 0,
                                     save_best_only = args.keep_best,
                                     mode = 'auto')

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_squared_error', optimizer=sgd)

        self.model.fit_generator(batch_gen(args.data_dir, x_train, y_train, args.batch_size, True),
                                 args.samples,
                                 args.num_epochs,
                                 max_q_size = 1,
                                 validation_data = batch_gen(args.data_dir, x_test, y_test, args.batch_size, False),
                                 nb_val_samples = len(x_test),
                                 callbacks = [self.checkpoint],
                                 verbose = 1)

    def build(self, args):
        self.model = Sequential()
        self.model.add(Conv2D(3, 1, 1, activation='elu', input_shape=second_shape))
        self.model.add(Conv2D(32, 3, 3, activation='elu'))
        self.model.add(Conv2D(32, 3, 3, activation='elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(64, 3, 3, activation='elu'))
        self.model.add(Conv2D(64, 3, 3, activation='elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(128, 3, 3, activation='elu'))
        self.model.add(Conv2D(128, 3, 3, activation='elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='elu'))
        self.model.add(Dense(64, activation='elu'))
        self.model.add(Dense(16, activation='elu'))
        self.model.add(Dense(1))


        self.model.summary()

class model_2(model):
    def train(self, args, x_train, x_valid, y_train, y_valid):
        self.checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                          monitor='val_loss',
                                          verbose=0,
                                          save_best_only=args.keep_best,
                                          mode='auto')

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss='mean_squared_error', optimizer=adam)

        self.model.fit_generator(batch_NVIDIA_gen(args.data_dir, x_train, y_train, args.batch_size, True),
                                 args.samples,
                                 args.num_epochs,
                                 max_q_size=1,
                                 validation_data=batch_NVIDIA_gen(args.data_dir, x_valid, y_valid, args.batch_size,
                                                                 False),
                                 nb_val_samples=len(x_valid),
                                 callbacks=[self.checkpoint],
                                 verbose=1)

    def build(self, args):
        self.model = Sequential()
        self.model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
        self.model.add(Conv2D(32, 8, 8, border_mode='same', subsample=(4, 4)))
        self.model.add(Conv2D(64, 8, 8, activation = 'relu', border_mode='same', subsample=(4, 4)))
        self.model.add(Conv2D(128, 4, 4, activation = 'relu',  border_mode='same', subsample=(2, 2)))
        self.model.add(Conv2D(128, 2, 2, activation = 'relu',  border_mode='same', subsample=(1, 1)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128))
        self.model.add(Dense(1))
        self.model.summary()

# Loading data from .csv file generated in training mode
def load_data(args):
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'),
                          names = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    Y = data_df['steering'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_size, random_state = 123)

    return X_train, X_test, Y_train, Y_test

def main():
    parse = argparse.ArgumentParser(description = 'Car driving with machine learning')
    parse.add_argument('-d', help = 'data directory', dest = 'data_dir', type = str, default = 'data')
    parse.add_argument('-t', help = 'test size fraction', dest = 'test_size', type = float, default = 0.25)
    parse.add_argument('-k', help = 'dropout probability', dest = 'keep_prob', type = float, default = 0.5)
    parse.add_argument('-n', help = 'number of epochs', dest = 'num_epochs', type = int, default = 10)
    parse.add_argument('-s', help = 'amount of samples per epoch', dest = 'samples', type = int, default = 20000)
    parse.add_argument('-b', help = 'batch size', dest = 'batch_size', type = int, default = 40)
    parse.add_argument('-o', help = 'keep best only', dest = 'keep_best', type = bool, default = True)
    parse.add_argument('-L', help = 'learning rate', dest = 'learn_rate', type = float, default = 1.0e-4)
    parse.add_argument('-0', help = 'NVIDIA car driving model', dest = 'NVIDIA_model', type = bool, default = True)
    parse.add_argument('-1', help = 'inny model 1', dest = 'inny_model1', type = bool, default = False)
    parse.add_argument('-2', help = 'inny model 2', dest = 'inny_model2', type = bool, default = False)
    args = parse.parse_args()

    data = load_data(args)

    if args.inny_model1:
        mod = model_1()
        print('model_1')
        args.NVIDIA_model = False
    elif args.inny_model2:
        mod = model_2()
        print('model_2')
        args.NVIDIA_model = False
    elif args.NVIDIA_model:
        mod = NVIDIA_model()
        print('NVIDIA')


    mod.build(args)
    mod.train(args, *data)

if __name__ == '__main__':
    main()
