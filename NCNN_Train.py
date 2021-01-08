from datetime import datetime
import os.path
import time
import numpy as np
import tensorflow as tf
import NCNN
import audio_reader

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

'''learning rate, could be fun to screw around with'''
LR = 0.00001

FLAGS = tf.compat.v1.flags.FLAGS

# storing checkpoints, the directories are placeholder, haven't tested them
tf.compat.v1.flags.DEFINE_string(
    'train_dir',
    './',
    """Directory for writing the event logs""")
tf.compat.v1.flags.DEFINE_string(
    'sum_dir',
    './summary',
    """Directory for writing summary""")
tf.compat.v1.flags.DEFINE_string(
    'noise_dir',
    './dataset/Noise_training',
    """Directory of noise files""")
tf.compat.v1.flags.DEFINE_string(
    'noisy_dir',
    './dataset/NoisySpeech_training',
    """Directory of noisy speech files""")
tf.compat.v1.flags.DEFINE_string(
    'clean_dir',
    './dataset/CleanSpeech_training',
    """Directory of clean speech files"""
)
'''add validation directories and validation dataset'''
tf.compat.v1.flags.DEFINE_string(
    'val_noisy_dir',
    './dataset/NoisySpeech_validation',
    """Directory of noisy speech files, for validating"""
)
tf.compat.v1.flags.DEFINE_string(
    'val_clean_dir',
    './dataset/CleanSpeech_validation',
    """Directory of clean speech files, for validating"""
)

tf.compat.v1.flags.DEFINE_integer('max_steps', 2000000000, """Number of batches to run""")

FFTP = 256  # number of fft points
EFTP = 129  # number of effective fft points
frame_move = 64  # roughly 8ms, amount by which
batch_size = 128
FRAME_IN = 8  # amount of spectogram frames presented to the net
FRAME_OUT = 1  # we want to have just one frame of a full length audio file going out
validation_samples = 700000  # we need to have a total validation set number here
batch_of_val = np.floor(validation_samples / batch_size)
val_left_to_dequeue = validation_samples - batch_of_val * batch_size
val_loss = np.zeros([1000000])


def train():
    coord = tf.train.Coordinator()

    audio_r = audio_reader.Audio_reader(
        FLAGS.clean_dir, FLAGS.noisy_dir, coord,
        FRAME_IN, FFTP, frame_move, is_validation=False)

    '''THIS WONT WORK, the directories for validation arent there
    we need to run the speech synthesizer with appropriate cfg to
    generate audio files for generation'''
    val_audio_r = audio_reader.Audio_reader(
        FLAGS.val_clean_dir, FLAGS.val_noisy_dir, coord,
        FRAME_IN, FFTP, frame_move, is_validation=False)

    # we should have a validation audio reader here for validating

    is_val = tf.compat.v1.placeholder(dtype=tf.bool, shape=())
    NoiseNET = NCNN.NoiseNet(batch_size, EFTP, FRAME_IN, FRAME_OUT)

    train_data_frames = audio_r.dequeue(batch_size)
    val_data_frames = val_audio_r.dequeue(batch_size)

    data_frames = tf.cond(is_val, lambda: val_data_frames, lambda: train_data_frames)

    images, targets = NoiseNET.inputs(data_frames)
    inf_targets = NoiseNET.inference(images, is_train=True)

    loss = NoiseNET.loss(inf_targets, targets)

    train_op = NoiseNET.train_optimizer(loss, LR)
    saver = tf.compat.v1.train.Saver(tf.compat.v1.all_variables())

    summary_op = tf.compat.v1.summary.merge_all()
    init = tf.compat.v1.initialize_all_variables()

    sess = tf.compat.v1.Session()

    sess.run(init)

    audio_r.start_threads(sess)
    val_audio_r.start_threads(sess)

    summary_writer = tf.compat.v1.summary.FileWriter(
        FLAGS.sum_dir,
        sess.graph
    )

    val_loss_id = 0

    for step in range(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run(
            [train_op, loss], feed_dict={is_val: False})

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        # training loss every 100 steps
        if step % 100 == 0:
            num_examples_per_step = batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = (
                '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
            print(format_str % (datetime.now(), step,
                                loss_value, examples_per_sec, sec_per_batch))
        # summary every 100 steps
        if step % 100 == 0:
            summary_str = sess.run(
                summary_op, feed_dict={is_val: False})
            summary_writer.add_summary(summary_str, step)

        # validation every 100000 step
        if step % 100000 == 0 or (step + 1) == FLAGS.max_steps:
            np_val_loss = 0
            print('Validation in progress...')
            for j in range(int(batch_of_val)):
                temp_loss = sess.run(
                    [loss], feed_dict={is_val: True})
                np_val_loss += temp_loss[0]
            val_audio_r.dequeue(val_left_to_dequeue)
            mean_val_loss = np_val_loss / batch_of_val
            print('validation loss %.2f' % mean_val_loss)
            val_loss[val_loss_id] = mean_val_loss
            val_loss_id += 1
            np.save('val_loss2.npy', val_loss)

        # store model every 10000 step
        if step % 10000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


train()