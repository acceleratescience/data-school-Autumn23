import numpy as np
import os

from numpy import ndarray
import torch
import torchaudio.transforms as T

from scipy.io.wavfile import read

from audio_classifier.summary_stats import SummaryStats

from audio_classifier.utils import load_audio_file

import unittest
import tempfile

class TestSummaryStats(unittest.TestCase):
    def setUp(self):
        # Create an instance of the SummaryStats class with default parameters
        self.summary_stats = SummaryStats()

    def test_fit(self):
        # Test that the fit method returns self
        self.assertIs(self.summary_stats.fit(None), self.summary_stats)

    def test_generate_mel_spectrogram(self):
        # Test the generate_mel_spectrogram method
        audio = np.random.rand(10000)  # Random audio data
        spectrogram = self.summary_stats.generate_mel_spectrogram(audio)
        self.assertTrue(isinstance(spectrogram, np.ndarray))
        self.assertEqual(spectrogram.shape[0], self.summary_stats.n_mels)

    def test_summary_statistics(self):
        # Test the summary_statistics method
        spectrogram = np.random.rand(self.summary_stats.n_mels, 100)  # Random spectrogram
        stats = self.summary_stats.summary_statistics(spectrogram)
        expected_length = 2 * self.summary_stats.n_mels
        self.assertTrue(isinstance(stats, np.ndarray))
        self.assertEqual(stats.shape[0], expected_length)

    def test_transform(self):
        # Test the transform method
        audio_list = [np.random.rand(10000) for _ in range(5)]  # List of random audio data
        transformed_data = self.summary_stats.transform(audio_list)
        self.assertTrue(isinstance(transformed_data, np.ndarray))
        self.assertEqual(transformed_data.shape[0], len(audio_list))
        expected_shape = (len(audio_list), 2 * self.summary_stats.n_mels)
        self.assertEqual(transformed_data.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()