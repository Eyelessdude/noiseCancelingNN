import os
import numpy as np
import pickle as pk
from scipy import signal
from scipy.io import wavfile

out_path = './pickle_data'
noisy_input_data_path = '../dataset/NoisySpeech_training'
clean_input_data_path = '../dataset/CleanSpeech_training'


def create_pickle_file(audio_path):
    temp_audio = []
    for dir_path, _, files in os.walk(audio_path):
        print('Dir path: ', dir_path)
        print('File name: ', files)

        for file in files:
            if file.endswith('.wav'):
                sample_rate, samples = wavfile.read(os.path.join(dir_path, file))
                sample_freq, sample_times, Zxx = signal.stft(samples, sample_rate, nperseg=256, nfft=256)
                times = len(sample_times) // 16

                for i in range(0, times * 16, 16):
                    temp_audio.append(np.log(np.abs(Zxx[:, i:i + 16]) + 1e-8))
    return temp_audio


NoisyAudio = create_pickle_file(noisy_input_data_path).copy()
CleanAudio = create_pickle_file(clean_input_data_path).copy()

# np.save('noisyAudio.npy', NoisyAudio)
# np.save('cleanAudio.npy', CleanAudio)
pickle_file = open(os.path.join(out_path, 'noisyAudio.pickle'), 'wb')
pk.dump(NoisyAudio, pickle_file)

pickle_file = open(os.path.join(out_path, 'cleanAudio.pickle'), 'wb')
pk.dump(CleanAudio, pickle_file)
