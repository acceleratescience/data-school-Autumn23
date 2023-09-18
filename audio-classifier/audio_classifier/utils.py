import numpy as np
import os

from scipy.io.wavfile import read


def load_audio_file(file_path):
    sr, data = read(file_path)
    return data, sr


def get_data(folder_path):
    labels = []
    audio_data = []
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            # Get spectrogram and summary statistics
            audio, _ = load_audio_file(os.path.join(folder_path, file))
            audio_data.append(audio)

            # Get labels
            label = int(file[:1])
            labels.append(label)

    return audio_data, np.array(labels)
