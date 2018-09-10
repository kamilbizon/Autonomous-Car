from image_processing import INPUT_SHAPE
from image_processing import batch_gen as batch_generator
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

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
                                     verbose = 0, save_best_only = args.keep_best,
                                     mode = 'auto')

        self.model.compile(loss = 'mean_squared_error', optimizer = Adam(lr = args.learn_rate))

        self.model.fit_generator(batch_generator(args.data_dir, x_train, y_train, args.batch_size, True),
                                 args.samples,
                                 args.num_epochs,
                                 max_q_size = 1,
                                 validation_data = batch_generator(args.data_dir, x_valid, y_valid, args.batch_size, False),
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
        

# jakbys mial jakies pomysly jakie modele zrobic to tutaj można napisać
class INNY_model_1(model):
    pass

class INNY_model_2(model):
    pass

# funkcja do ładowania 
def load_data(args):
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'),
                          names = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    Y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid

def main():
    parse = argparse.ArgumentParser(description = 'Car driving with machine learning')
    parse.add_argument('-d', help = 'data directory', dest = 'data_dir', type = str, default = 'data')
    parse.add_argument('-t', help = 'test size fraction', dest = 'test_size', type = float, default = 0.2)
    parse.add_argument('-k', help = 'dropout probability', dest = 'keep_prob', type = float, default = 0.5)
    parse.add_argument('-n', help = 'number of epochs', dest = 'num_epochs', type = int, default = 10)
    parse.add_argument('-s', help = 'amount of samples per epoch', dest = 'samples', type = int, default = 20000)
    parse.add_argument('-b', help = 'batch size', dest = 'batch_size', type = int, default = 40)
    parse.add_argument('-o', help = 'keep best only', dest = 'keep_best', type = bool, default = True)
    parse.add_argument('-l', help = 'learning rate', dest = 'learn_rate', type = float, default = 1.0e-4)
    parse.add_argument('-1', help = 'NVIDIA car driving model', dest = 'NVIDIA_model', type = bool, default = True)
    parse.add_argument('-2', help = 'inny model 1', dest = 'inny_model1', type = bool, default = False)
    parse.add_argument('-3', help = 'inny model 2', dest = 'inny_model2', type = bool, default = False)
    args = parse.parse_args()

    data = load_data(args)
    
    mod = NVIDIA_model()
    mod.build(args)
    mod.train(args, *data)

if __name__ == '__main__':
    main()
