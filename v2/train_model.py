import os
import tensorflow as tf
import pickle as pk
import numpy as np
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam


def get_gpus():
    local_devices = device_lib.list_local_devices()
    gpus = []
    for device in local_devices:
        if device.device_type == 'GPU':
            gpus.append(device)
    return gpus


print(get_gpus())

# Add number of your device here!
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

main_path = os.path.dirname(os.path.abspath(__file__))
noisy_pickle_path = '/pickle_data/noisyAudio.pickle'
clean_pickle_path = '/pickle_data/cleanAudio.pickle'
epochs = 10
batch_size = 64
for device in ['LIST OF CUDA DEVICES']:
    with tf.device(device):
        tf.config.experimental.set_memory_growth(device, True)
        noisyPickle = open(main_path + noisy_pickle_path, 'rb')
        x_train = pk.load(noisyPickle)
        cleanPickle = open(main_path + clean_pickle_path, 'rb')
        y_train = pk.load(cleanPickle)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_train = x_train.reshape(x_train.shape[0], 129, 16, 1)
        y_train = y_train.reshape(y_train.shape[0], 129, 16, 1)

        # X_train = []
        #
        # for i in range(7, len(x_train)):
        #     X_train.append(x_train[i - 7:i + 1])
        #
        # X_train = np.asarray(X_train)
        # y_train = y_train.reshape(y_train.shape[0], 1, 129, 16)

        model = Sequential()
        model.add(Conv2D(64, kernel_size=(7, 7), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', input_shape=(129, 16, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, kernel_size=(1, 1), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(128, (3, 3), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(64, (5, 5), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(1, (7, 7), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.compile(loss=mean_squared_error, optimizer=Adam(), metrics=['accuracy'])

        print(model.summary())

        model.fit(np.asarray(x_train), np.asarray(y_train), batch_size=batch_size, epochs=epochs,
                  validation_split=0.01)

        model.save_weights('../weights.h5')
        model.save('../model.h5')
