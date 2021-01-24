'''
Evaluate a trained model using a noisy speech
'''

import tensorflow as tf
import numpy as np
import soundfile as sf
import librosa
import NCNN
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks
from tensorflow.python.framework.ops import disable_eager_execution

log10_fac = 1 / np.log(10)
disable_eager_execution()

FLAGS = tf.compat.v1.flags.FLAGS
# directory to load the model
tf.compat.v1.flags.DEFINE_string(
    'train_dir',
    './models',
    """Model directory""")

LR = 0.0001
FRAME_IN = 8
EFTP = 129
FFTP = 256
FRAME_OUT = 1
mul_fac = 0.2
frame_move = 64
noisy_dir = './dataset/SNR10/NoisySpeech_validation/noisy80.wav'
clean_dir = './dataset/SNR10/CleanSpeech_validation/clnsp80.wav'
out_original_noisy_dir = './validation/test_noisy.wav'
out_original_clean_dir = './validation/test_clean.wav'
out_audio_dir = './validation/test_NCNN.wav'

noisy_org, sr = librosa.load(noisy_dir, sr=None)
clean_org, _ = librosa.load(clean_dir, sr=None)


in_stft = np.transpose(librosa.core.stft(noisy_org, n_fft=FFTP, hop_length=frame_move, window=np.hanning(FFTP)), [1, 0])
in_stft_amp = np.maximum(np.abs(in_stft), 1e-5)
in_data = 20. * np.log10(in_stft_amp * 100)
phase_data = in_stft / in_stft_amp

data_len = in_data.shape[0]
assert EFTP == in_data.shape[1], 'Image height incompatible'
out_len = data_len - FRAME_IN + 1
shape = int((out_len - 1) * frame_move + FFTP)
out_audio = np.zeros(shape=[shape])


init_op = tf.compat.v1.initialize_all_variables()

batch_size = 1

NoiseNet = NCNN.NoiseNet(batch_size, EFTP, FRAME_IN, FRAME_OUT)

images = tf.compat.v1.placeholder(tf.float32, [FRAME_IN, EFTP])

targets = tf.compat.v1.placeholder(tf.float32, [EFTP])

inf_targets = NoiseNet.inference(images, is_train=True)

loss = NoiseNet.loss(inf_targets, targets)

# train_op = SE_Net.train(loss, LR)

# saver = tf.compat.v1.train.import_meta_graph('./models/model.ckpt-10000.meta')
saver = tf.compat.v1.train.Saver(tf.compat.v1.all_variables())
sess = tf.compat.v1.Session()
summary_op = tf.compat.v1.summary.merge_all()

population_mean = tf.compat.v1.placeholder(tf.float32)
population_var = tf.compat.v1.placeholder(tf.float32)


# with tf.compat.v1.Session() as sess:
# restore the model
model_name = './models/21-01-2021 - SNR10 janky/model-1900000'
saver.restore(sess, model_name)
# sess.run(tf.compat.v1.initialize_all_variables())
print("Model restored: %s" % model_name)
i = 0
while(i < out_len):
    # show progress
    if i % 100 == 0:
        print('frame: %d' % (i))
    feed_in_data = in_data[i:i + FRAME_IN][:] #data loaded from audio file
    # normalization
    data_mean = np.mean(feed_in_data) #data from saved model
    data_var = np.var(feed_in_data) #data from saved model
    feed_in_data = (feed_in_data - data_mean) / np.sqrt(data_var)
    # get the speech inference
    inf_frame = np.asarray(sess.run(
        inf_targets,
        feed_dict={images: feed_in_data}))
    inf_frame = inf_frame * np.sqrt(data_var) + data_mean
    out_amp_tmp = 10 ** (inf_frame / 20) / 100
    out_stft = out_amp_tmp * phase_data[i + FRAME_IN - 1][:]
    out_stft.shape = (EFTP, )
    con_data = out_stft[-2:0:-1].conjugate()
    out_amp = np.concatenate((out_stft, con_data))
    frame_out_tmp = np.fft.ifft(out_amp).astype(np.float64)
    # frame_out_tmp = frame_out_tmp / 255
    # overlap and add to get the final time domain wavform
    slice_start = int(i * frame_move)
    slice_end = int(i * frame_move + FFTP)
    out_audio[slice_start: slice_end] += frame_out_tmp * 0.5016
    i = i + 1
    # data_f = 10 * tf.compat.v1.math.log(data_f5 * 10000) * log10_fac
# length = img.shape[]

# ipdb.set_trace()
# store the computed results
sf.write(out_audio_dir, out_audio, sr)
sf.write(out_original_noisy_dir, noisy_org, sr)
sf.write(out_original_clean_dir, clean_org, sr)
