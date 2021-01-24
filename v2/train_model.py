import os
import tensorflow as tf
import pickle as pk
import numpy as np
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import load_model
import lr_reducer


def get_gpus():
    local_devices = device_lib.list_local_devices()
    gpus = []
    for device in local_devices:
        if device.device_type == 'GPU':
            gpus.append(device)
    return gpus


print(get_gpus())

# Add number of your device here!


main_path = os.path.dirname(os.path.abspath(__file__))
noisy_pickle_path = '/pickle_data/noisyAudio.pickle'
clean_pickle_path = '/pickle_data/cleanAudio.pickle'
root_path = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = '/checkpoints'

epochs = 240
batch_size = 64
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# for device in ['/device:GPU:0']:
# with tf.device(device):
# tf.config.experimental.set_memory_growth(device, True)
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

# model = load_model(root_path + '/model.h5')



# Layer 1


def conv2dmodel(SGD_op=False):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 7), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', input_shape=(129, 16, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 2
    model.add(Conv2D(128, kernel_size=(5, 5), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 3
    model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 4
    model.add(Conv2D(256, kernel_size=(1, 1), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 5
    model.add(Conv2DTranspose(128, (3, 3), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 6
    model.add(Conv2DTranspose(64, (5, 5), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 7
    model.add(Conv2DTranspose(1, (7, 7), padding='valid', use_bias=True, kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=main_path + checkpoint_path,
    #     save_weights_only=True,
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True,
    #     save_freq=13010)

    if not SGD_op:
        model.compile(loss=mean_squared_error, optimizer=optimizers.Adam(), metrics=['accuracy'])
        print(model.summary())
        model.fit(np.asarray(x_train), np.asarray(y_train), batch_size=batch_size, epochs=epochs, validation_split=0.01)
                  # , callbacks=[model_checkpoint_callback])
    else:
        model.compile(loss=mean_squared_error, optimizer=optimizers.SGD(), metrics=['accuracy'])
        print(model.summary())
        model.fit(np.asarray(x_train), np.asarray(y_train), batch_size=batch_size, epochs=epochs,
                  validation_split=0.01) # , callbacks=[model_checkpoint_callback]) # , callbacks=[lr_reducer.LearningRateReduce(rate=0.99)])

    if not SGD_op:
        model.save_weights('../weightsconv2d-AdamOpt.h5')
        model.save('../modelconv2d-AdamOpt.h5')
    else:
        model.save_weights('../weightsconv2d-SGDOpt.h5')
        model.save('../modelconv2d-SGDOpt.h5')


def conv1dmodel(SGD_op=False):
    model = Sequential()
    # Layer 1
    model.add(Conv2D(12, kernel_size=[13, 1], padding='valid', use_bias=True,
                     bias_initializer='zeros', input_shape=(129, 16, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 2
    model.add(Conv2D(16, kernel_size=[11, 1], padding='valid', use_bias=True,
                     bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 3
    model.add(Conv2D(20, kernel_size=[9, 1], padding='valid', use_bias=True,
                     bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 4
    model.add(Conv2D(24, kernel_size=[7, 1], padding='valid', use_bias=True,
                     bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 5
    model.add(Conv2D(32, kernel_size=[5, 1], padding='valid', use_bias=True,
                     bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(32, kernel_size=[5, 1], padding='valid', use_bias=True,
                              bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 6
    model.add(Conv2DTranspose(24, kernel_size=[7, 1], padding='valid', use_bias=True,
                              bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 7
    model.add(Conv2DTranspose(20, kernel_size=[9, 1], padding='valid', use_bias=True,
                              bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 8
    model.add(Conv2DTranspose(16, kernel_size=[11, 1], padding='valid', use_bias=True,
                              bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 9
    model.add(Conv2DTranspose(12, kernel_size=[13, 1], padding='valid', use_bias=True,
                              bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 10
    model.add(Conv2DTranspose(1, kernel_size=[1, 1], padding='valid', use_bias=True,
                              bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 11
    # model.add(Conv2DTranspose(1, kernel_size=[25, 1], padding='valid', use_bias=True,
    #                           bias_initializer='zeros'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=main_path + checkpoint_path,
    #     save_weights_only=True,
    #     monitor='val_loss',
    #     mode='min',
    #     save_freq=13010)

    if not SGD_op:
        model.compile(loss=mean_squared_error, optimizer=optimizers.Adam(), metrics=['accuracy'])

        print(model.summary())

        model.fit(np.asarray(x_train), np.asarray(y_train), batch_size=batch_size, epochs=epochs,
                  validation_split=0.01) # , callbacks=[model_checkpoint_callback])  # , callbacks=[lr_reducer.LearningRateReduce(rate=0.999)]
    else:
        model.compile(loss=mean_squared_error, optimizer=optimizers.SGD(), metrics=['accuracy'])

        print(model.summary())
        model.fit(np.asarray(x_train), np.asarray(y_train), batch_size=batch_size, epochs=epochs,
                  validation_split=0.01) # , callbacks=[model_checkpoint_callback]) # , callbacks=[lr_reducer.LearningRateReduce(rate=0.99)])

    if not SGD_op:
        model.save_weights('../weightsconv1d-AdamOpt.h5')
        model.save('../modelconv1d-AdamOpt.h5')
    else:
        model.save_weights('../weightsconv1d-SGDOpt.h5')
        model.save('../modelconv1d-SGDOpt.h5')

# print('conv1dmodelSGD start\n')
# conv1dmodel(SGD_op=True)
print('conv1dmodelAdam start\n')
conv1dmodel()
print('conv2dmodelSGD start\n')
conv2dmodel(SGD_op=True)
# low priority - i already ran that
# print('conv2dmodelAdam start\n')
# conv2dmodel()

