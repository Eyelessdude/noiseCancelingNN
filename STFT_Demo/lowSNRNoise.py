import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "../v2/test_noisy.wav"

signal, sr = librosa.load(file, sr=16000)

# Display the wave plot
librosa.display.waveplot(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# Display the spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)

# Linspace gives us number of evenly spaced numbers in the intervals
frequency = np.linspace(0, sr, len(magnitude))

# Get only first half as they are mirrored
frequency_part = frequency[:int(len(frequency) / 2)]
magnitude_part = magnitude[:int(len(magnitude) / 2)]

plt.plot(frequency_part, magnitude_part)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

# Apply STFT and get spectrogram
fft_windows = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=fft_windows)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()