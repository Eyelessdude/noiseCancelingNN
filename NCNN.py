import tensorflow as tf
import numpy as np

log10_fac = 1 / np.log(10)


def var_sum(var):
    with tf.name_scope('summaries'):
        tensor_name = var.op.name
        mean = tf.reduce_mean(var)
        tf.compat.v1.summary.scalar(tensor_name + 'mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.compat.v1.summary.scalar(tensor_name + 'stddev', stddev)
        tf.compat.v1.summary.scalar(tensor_name + 'max', tf.reduce_max(var))
        tf.compat.v1.summary.scalar(tensor_name + 'min', tf.reduce_min(var))
        tf.compat.v1.summary.histogram(tensor_name + 'histogram', var)


def conv1d(x, W):
    return tf.compat.v1.nn.conv2d(x, W, strides=[1, 128, 1, 1], padding='SAME')
    # for convenience 2d function is used
    # it works like a 1d convolution does - a single long vector. change 128 to parameter: self.EFTP later


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class NoiseNet(object):
    def __init__(self, batch_size, EFTP, FRAME_IN, FRAME_OUT, DECAY=0.999):
        self.batch_size = batch_size
        self.EFTP = EFTP
        self.FRAME_IN = FRAME_IN
        self.FRAME_OUT = FRAME_OUT
        self.DECAY = DECAY  # global mean and var estimation using batchnorm decay

    def inputs(self, data_frames):  # it's mostly the fourier transform
        data_frames_t = tf.compat.v1.transpose(data_frames, perm=[2, 0, 1, 3]) #shape : 2, gen_frame, frame_in, frame_len
        raw_noisyspeech = data_frames_t[0][:][:][:]
        raw_cleanspeech = data_frames_t[1][:][:][:]

        # Fast fourier transform
        noisy_speech = tf.compat.v1.signal.fft(tf.cast(raw_noisyspeech, tf.complex64))
        noisy_speech = tf.compat.v1.transpose(noisy_speech, [2, 0, 1])
        #limit FFTP to EFTP
        noisy_speech = noisy_speech[0:self.EFTP][:][:]
        noisy_speech = tf.compat.v1.transpose(noisy_speech, [1, 2, 0])

        clean_speech = tf.compat.v1.signal.fft(tf.cast(raw_cleanspeech, tf.complex64))
        clean_speech = tf.compat.v1.transpose(clean_speech, [2, 0, 1])
        clean_speech = clean_speech[0:self.EFTP][:][:]
        clean_speech = tf.compat.v1.transpose(clean_speech, [1, 2, 0])

        # shape:
        # batch, N_in, NEFF
        images = tf.cast(noisy_speech, tf.float32)
        targets = [tf.reshape(tf.cast(clean_speech, tf.float32)[i][self.FRAME_IN - 1][0:self.EFTP], [1, self.EFTP])
                   for i in range(0, self.batch_size, 1)]

        # do per image whitening
        images_reshape = tf.transpose(tf.reshape(
            images, [self.batch_size, -1]))
        targets_reshape = tf.transpose(tf.reshape(
            targets, [self.batch_size, -1]))
        batch_mean, batch_var = tf.nn.moments(images_reshape, [0])
        images_reshape_norm = tf.compat.v1.nn.batch_normalization(
            images_reshape, batch_mean, batch_var, 0, 1, 1e-10)
        targets_reshape_norm = tf.compat.v1.nn.batch_normalization(
            targets_reshape, batch_mean, batch_var, 0, 1, 1e-10)
        images_norm = tf.reshape(tf.transpose(images_reshape_norm),
                                 [self.batch_size, self.FRAME_IN, self.EFTP])
        targets_norm = tf.reshape(tf.transpose(targets_reshape_norm),
                                  [self.batch_size, self.EFTP])
        return images_norm, targets_norm

    def batch_norm_wrapper(self, inputs, is_training, epsilon=1e-6):
        scale = tf.Variable(tf.ones(inputs.get_shape()[-1]))
        beta = tf.Variable(tf.ones(inputs.get_shape()[-1]))

        population_mean = tf.Variable(
            tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        population_var = tf.Variable(
            tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training:
            batch_mean, batch_var = tf.compat.v1.nn.moments(inputs, [0, 1, 2])
            train_mean = tf.compat.v1.assign(population_mean,
                                             population_mean * self.DECAY + batch_mean * (1 - self.DECAY))
            train_var = tf.compat.v1.assign(population_var,
                                            population_var * self.DECAY + batch_mean * (1 - self.DECAY))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.compat.v1.nn.batch_normalization(
                    inputs, batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.compat.v1.nn.batch_normalization(
                inputs, population_mean, population_var, beta, scale, epsilon)

    def conv_layer_wrapper(self, input, out_feature_maps, filter_length, is_train):
        filter_width = input.get_shape()[1]
        in_feature_maps = input.get_shape()[-1]
        W_conv = weight_variable(
            [filter_width, filter_length, in_feature_maps, out_feature_maps]
        )
        b_conv = bias_variable([out_feature_maps])
        h_conv_t = conv1d(input, W_conv)
        # batch normalisation
        h_conv_b = self.batch_norm_wrapper(h_conv_t, is_train)
        return tf.compat.v1.nn.relu(h_conv_b)

    def inference(self, images, is_train):
        # a lot of stuff we could mess around in here - this is where we basically
        # set up the layers. so a lot of fun stuff. for now filter widths, filter
        # number etc. are as in the paper
        image_input = tf.reshape(images, [-1, self.FRAME_IN, self.EFTP, 1])
        with tf.compat.v1.variable_scope('layer1') as scope:
            h_layer1 = self.conv_layer_wrapper(image_input, 12, 13, is_train)
        with tf.compat.v1.variable_scope('layer2') as scope:
            h_layer2 = self.conv_layer_wrapper(h_layer1, 16, 11, is_train)
        with tf.compat.v1.variable_scope('layer3') as scope:
            h_layer3 = self.conv_layer_wrapper(h_layer2, 20, 9, is_train)
        with tf.compat.v1.variable_scope('layer4') as scope:
            h_layer4 = self.conv_layer_wrapper(h_layer3, 24, 7, is_train)
        with tf.compat.v1.variable_scope('layer5') as scope:
            h_layer5 = self.conv_layer_wrapper(h_layer4, 32, 5, is_train)
        with tf.compat.v1.variable_scope('layer6') as scope:
            h_layer6 = self.conv_layer_wrapper(h_layer5, 24, 7, is_train)
        with tf.compat.v1.variable_scope('layer7') as scope:
            h_layer7 = self.conv_layer_wrapper(h_layer6, 20, 9, is_train)
        with tf.compat.v1.variable_scope('layer8') as scope:
            h_layer8 = self.conv_layer_wrapper(h_layer7, 16, 11, is_train)
        with tf.compat.v1.variable_scope('layer9') as scope:
            h_layer9 = self.conv_layer_wrapper(h_layer8, 12, 13, is_train)
        with tf.compat.v1.variable_scope('layer10') as scope:
            f_w = h_layer9.get_shape()[1]
            i_fm = h_layer9.get_shape()[-1]
            W_conv10 = weight_variable(
                [f_w, 129, i_fm, 1]
            )
            b_conv10 = bias_variable([1])
            h_layer10 = conv1d(h_layer9, W_conv10) + b_conv10
        return tf.reshape(h_layer10, [-1, self.EFTP])

    def loss(self, inf_targets, targets):
        loss_v = tf.compat.v1.nn.l2_loss(inf_targets - targets) / self.batch_size
        tf.compat.v1.summary.scalar('loss', loss_v)
        return loss_v

    def train_optimizer(self, loss, lr):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss)
        return train_op
