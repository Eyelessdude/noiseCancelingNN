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
    def __init__(self, sound_dir, coord, FRAME_IN, frame_length, frame_move, is_validation):
        self.sound_dir = sound_dir
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
        self.soundfiles = find_files(sound_dir)
        print('%d sound files found' % len(self.soundfiles))

    def dequeue(self, num_elements):
        output = self.q.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        stop = False

        len_sound_files = len(self.soundfiles)
        count = 0
        while not stop:
            ids = list(range(len_sound_files))
            random.shuffle(ids)
            for i in ids:
                sound, _ = librosa.load(self.soundfiles[i], sr=None)
                '''number of generated frames'''
                gen_frames = np.floor(
                    (len(sound) - self.frame_length) / self.frame_move - self.FRAME_IN)
                data = sound
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
