import unittest
import os
import numpy as np
from audio_classifier.utils import load_audio_file, get_data  # Import your functions from your actual module
from scipy.io import wavfile

class TestAudioFunctions(unittest.TestCase):
    def setUp(self):
        # Create a temporary folder with test audio files for testing
        self.temp_folder = 'temp_test_audio'
        os.makedirs(self.temp_folder, exist_ok=True)   


    def tearDown(self):
        # Clean up the temporary folder and audio files after testing
        if os.path.exists(self.temp_folder):
            for file in os.listdir(self.temp_folder):
                file_path = os.path.join(self.temp_folder, file)
                os.remove(file_path)
            os.rmdir(self.temp_folder)


    def test_load_audio_file(self):
        # Test the load_audio_file function
        sample_audio_path = os.path.join(self.temp_folder, 'sample_audio.wav')
        sample_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        wavfile.write(sample_audio_path, 44100, sample_data)
        data, sr = load_audio_file(sample_audio_path)

        self.assertTrue(isinstance(data, np.ndarray))
        self.assertEqual(sr, 44100)
        self.assertListEqual(list(data), list(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)))
        os.remove(sample_audio_path)


    def test_get_data(self):
        # Test the get_data function
        # Create test audio files with known labels
        labels = [1, 2, 3]
        for label in labels:
            audio_path = os.path.join(self.temp_folder, f'{label}_test_audio.wav')
            sample_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            wavfile.write(audio_path, 44100, sample_data)

        audio_data, label_data = get_data(self.temp_folder)
        self.assertTrue(isinstance(audio_data, list))
        self.assertTrue(isinstance(label_data, np.ndarray))
        self.assertEqual(len(audio_data), len(labels))
        self.assertListEqual(sorted(label_data.tolist()), labels)


if __name__ == '__main__':
    unittest.main()
