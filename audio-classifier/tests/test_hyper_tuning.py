import unittest
import numpy as np

# from audio_classifier.utils import SummaryStats, get_data, load_audio_file
from audio_classifier.tune import get_train_test_split, evaluate, tune

from sklearn.neighbors import KNeighborsClassifier

import os

from scipy.io.wavfile import write

class TestAudioFunctions(unittest.TestCase):
    def setUp(self):
        # Create a temporary folder with test audio files for testing
        self.temp_folder = 'temp_test_audio'
        os.makedirs(self.temp_folder, exist_ok=True)

        # Create three samples for each label in the range [0, 9]
        for label in range(10):
            for _ in range(10):
                audio_path = os.path.join(self.temp_folder, f'{label}_sample{_}.wav')
                # Create a sample audio waveform (you can replace this with actual audio data)
                sample_data = np.random.rand(44100)  # Random audio data with a sample rate of 44100
                write(audio_path, 44100, sample_data)

    def tearDown(self):
        # Clean up the temporary folder and audio files after testing
        if os.path.exists(self.temp_folder):
            for file in os.listdir(self.temp_folder):
                file_path = os.path.join(self.temp_folder, file)
                os.remove(file_path)
            os.rmdir(self.temp_folder)

    def test_get_train_test_split(self):
        # Test the get_train_test_split function
        X_train, X_test, y_train, y_test = get_train_test_split(self.temp_folder)
        self.assertTrue(isinstance(X_train, list))
        self.assertTrue(isinstance(X_test, list))
        self.assertTrue(isinstance(y_train, np.ndarray))
        self.assertTrue(isinstance(y_test, np.ndarray))
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

    def test_evaluate(self):
    # Test the evaluate function with a simple model
        model = KNeighborsClassifier(n_neighbors=1)  # Use a simple classifier for testing
        X_train, X_test, y_train, y_test = get_train_test_split(self.temp_folder)
        
        # Fit the model with training data before evaluating
        model.fit(X_train, y_train)
        
        y_pred, accuracy = evaluate(X_test, y_test, model)
        self.assertTrue(isinstance(y_pred, np.ndarray))
        self.assertTrue(isinstance(accuracy, float))
        self.assertEqual(len(y_pred), len(y_test))

    def test_tune(self):
        # Test the tune function with a simple parameter grid
        param_grid = {
            "classifier__n_neighbors": [1, 3, 5],
            "classifier__weights": ["uniform", "distance"],
        }
        X_train, _, y_train, _ = get_train_test_split(self.temp_folder)
        tune(X_train, y_train, param_grid, "best_model.joblib")
        self.assertTrue(os.path.exists("best_model.joblib"))

if __name__ == '__main__':
    unittest.main()
