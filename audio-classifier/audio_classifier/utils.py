
import numpy as np
import os

from numpy import ndarray
import torch
import torchaudio.transforms as T

from scipy.io.wavfile import read

from sklearn.base import BaseEstimator, TransformerMixin


class SummaryStats(BaseEstimator, TransformerMixin):
    def __init__(self, sr : int = 4000,
                 n_fft : int = 1024,
                 hop_length : int = 128,
                 n_mels : int = 40):
        """Generates spectrograms and summary statistics of audio data.

        Args:
            sr (int, optional): sample rate. Defaults to 4000.
            n_fft (int, optional): number of fft components. Defaults to 1024.
            hop_length (int, optional): hop length. Defaults to 128.
            n_mels (int, optional): number of mels. Defaults to 128.
        """
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels


    def fit(self, X, y=None):
        return self


    def transform(self, X : list) -> ndarray:
        """Transform input data

        Args:
            X (list): List of audio streams.

        Returns:
            ndarray: Summary statistics of audio streams.
        """
        data = []
        for x in X:
            # Get spectrogram and summary statistics
            log_S = self.generate_mel_spectrogram(x)
            stats = self.summary_statistics(log_S)
            data.append(stats)
        return np.array(data)


    def generate_mel_spectrogram(self, audio : ndarray) -> ndarray:
        """Gets a mel spectrogram from audio data

        Args:
            audio (ndarray): a 1D audio stream.

        Returns:
            ndarray: Spectrogram
        """
        transform = T.MelSpectrogram(self.sr, normalized=True)
        S = transform(torch.from_numpy(audio).type(torch.float))
        S = 10 * torch.log10(S)
        return S.numpy()
    

    def summary_statistics(self, spectrogram : ndarray) -> ndarray:
        """Get mean and standard deviation of each frequency row.

        Args:
            spectrogram (np.ndarray): A spectrogram

        Returns:
            ndarray: An array of size 2*n_mels.
        """
        mean = spectrogram.mean(axis=1)
        std = spectrogram.std(axis=1)
        stats = np.concatenate([mean, std])

        return stats
    

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

