'''Class reading the audiofiles. Transforms them from audio to readable
strides format'''

import tensorflow as tf
import librosa
import threading
import numpy as np
import fnmatch
import os
import random
from numpy.lib import stride_tricks


def find_files(directory, pattern='*.wav'):
    files = []
    for root, directory_name, filenames in os.walk(directory):
        for file in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, file))
    return files


class Audio_reader(object):
    def __init__(self, clean_dir, noisy_dir, coord, FRAME_IN, frame_length, frame_move, is_validation):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.coord = coord
        self.FRAME_IN = FRAME_IN
        self.frame_length = frame_length
        self.frame_move = frame_move
        self.is_validation = is_validation
        self.sample_placeholder_many = tf.compat.v1.placeholder(tf.float32,
                                                                shape=(None, self.FRAME_IN, 2, frame_length))

        if not is_validation:
            self.q = tf.queue.RandomShuffleQueue(200000, 5000, tf.float32, shapes=(self.FRAME_IN, 2, frame_length))
        else:
            self.q = tf.queue.FIFOQueue(200000, tf.float32, shapes=(self.FRAME_IN, 2, frame_length))
        self.enqueue_many = self.q.enqueue_many(self.sample_placeholder_many + 0)
        self.cleanfiles = find_files(clean_dir)
        self.noisyfiles = find_files(noisy_dir)
        print('%d speech found' % len(self.cleanfiles))
        print('%d noise found' % len(self.noisyfiles))

    def dequeue(self, num_elements):
        output = self.q.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        stop = False

        len_clean_files = len(self.cleanfiles)
        len_noisy_files = len(self.noisyfiles)
        count = 0
        while not stop:
            ids = list(range(len_clean_files))
            random.shuffle(ids)
            for i in ids:
                clean, _ = librosa.load(self.cleanfiles[i], sr=None)
                noisy, _ = librosa.load(self.noisyfiles[i], sr=None)
                '''number of generated frames'''
                gen_frames = np.floor(
                    (len(clean) - self.frame_length) / self.frame_move - self.FRAME_IN)
                data = np.concatenate((noisy, clean))
                stride = data.strides
                data_frames = stride_tricks.as_strided(
                    data,
                    shape=(gen_frames.astype(int), self.FRAME_IN, 2, self.frame_length),
                    strides=(
                        data.strides[0] * self.frame_move,
                        data.strides[0] * self.frame_move,
                        data.strides[0],
                        data.strides[0]
                    )
                )
                sess.run(self.enqueue_many,
                         feed_dict={self.sample_placeholder_many: data_frames})
                count += gen_frames
            np.save('sampleN.npy', count)

    def start_threads(self, sess, num_thread=1):
        for i in range(num_thread):
            thread = threading.Thread(
                target=self.thread_main, args=(sess,))
            thread.daemon = True
            thread.start()
