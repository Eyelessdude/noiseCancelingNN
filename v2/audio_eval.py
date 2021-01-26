import os
import numpy as np
# import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from tensorflow.keras.models import load_model

root_path = os.path.dirname(os.path.abspath(__file__))
# Replace with proper path.

# for j in range(1, 50):

tested_audio = '../dataset/SNR10/NoisySpeech_validation/home4.wav'

# load the model here
model = load_model(root_path + '/modelconv1d-AdamOpt.h5')

sample_rate, samples = wavfile.read(tested_audio)
NoiseAudio = []
Phase = []
sample_freq, sample_times, Zxx = signal.stft(samples, sample_rate, nperseg=256, nfft=256)
phases = np.angle(Zxx)
times = len(sample_times) // 16

for i in range(0, times * 16, 16):
    NoiseAudio.append(np.log(np.abs(Zxx[:, i:i + 16]) + 1e-8))
    Phase.append(phases[:, i:i + 16])

if len(sample_times) % 16 != 0:
    NoiseAudio.append(np.log(np.abs(Zxx[:, len(sample_times) - 16: len(sample_times)]) + 1e-8))
    Phase.append(phases[:, len(sample_times) - 16: len(sample_times)])
    times = times + 1
NoiseAudio = np.asarray(NoiseAudio)
NoiseAudio = NoiseAudio.reshape(NoiseAudio.shape[0], 129, 16, 1)
NoiseAudio = np.asarray(NoiseAudio)

result = model.predict(NoiseAudio)
result = result.reshape(result.shape[0], 129, 16)
result_Zxx = np.zeros((129, 0))

for i in range(times):
    exponential = np.exp(result[i])
    result_Zxx = np.append(result_Zxx, exponential * np.cos(Phase[i]) + 1j * exponential * np.sin(Phase[i]), axis=1)
signal_times, istft = signal.istft(result_Zxx, nperseg=256, nfft=256)
istft = np.asarray(istft, dtype=np.int16)

wavfile.write(root_path + '/result_mine/test_NCNN.wav', sample_rate, istft)
