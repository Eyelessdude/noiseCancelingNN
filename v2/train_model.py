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

noisy_pickle = './pickle_data/noisyAudio.pickle'
clean_pickle = './pickle_data/cleanAudio.pickle'
epochs = 10
batch_size = 64

for device in ['LIST OF CUDA DEVICES']:
    with tf.device(device):
        tf.config.experimental.set_memory_growth(device, True)

        noisyPickle = open('%s' % noisy_pickle, 'rb')
        x_train = pk.load(noisyPickle)
        cleanPickle = open('%s' % clean_pickle, 'rb')
        y_train = pk.load(cleanPickle)

        y_train = np.asarray(y_train)

        X_train = []

        for i in range(7, len(x_train)):
            X_train.append(x_train[i - 7:i + 1])

        X_train = np.asarray(X_train)
        y_train = y_train.reshape(y_train.shape[0], 1, 129, 16)

        model = Sequential()
        model.add(Conv2D(64, kernel_size=(7, 7), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', data_format='channels_first', input_shape=(8, 129, 16)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', data_format='channels_first'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', data_format='channels_first'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, kernel_size=(1, 1), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', data_format='channels_first'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(128, (3, 3), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros', data_format="channels_first"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(64, (5, 5), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros', data_format="channels_first"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(1, (7, 7), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros', data_format="channels_first"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])

        print(model.summary())

        with tf.device(device) as open:
            model.fit(np.asarray(X_train), np.asarray(y_train[7:]), batch_size=batch_size, epochs=epochs,
                      validation_split=0.01)

        model.save_weights('weights.h5')
        model.save('model.h5')
