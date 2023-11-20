!pip install hmmlearn
!pip install python_speech_features

import os
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from hmmlearn import hmm
from google.colab import drive
drive.mount('/content/drive')

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM classifier')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
                        help="Input folder containing the audio files in your Google Drive.")
    return parser

if __name__ == '__main__':
    data_dir = '/content/drive/MyDrive/Colab Notebooks/hmm-speech-recognition-0.1'

    # List all audio files in the data directory
    all_audio_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                all_audio_files.append(os.path.join(root, file))

    # Use all audio files for training
    training_files = all_audio_files

    # Initialize a list to store feature data and labels
    X = []  # List to store feature data
    y_words = []  # List to store class labels

    # Extract feature data from training files and store it in X and y_words
    for training_file in training_files:
        label = training_file.split("/")[-2]  # Extract the class label from the file path
        sampling_freq, audio = wavfile.read(training_file)
        mfcc_features = mfcc(audio, sampling_freq)
        X.append(mfcc_features)
        y_words.append(label)

    # Initialize a list to store HMM models
    hmm_models = []

    # Define and use the HMMTrainer class
    class HMMTrainer:
        def __init__(self, n_components=3, n_iter=1000):
            self.n_components = n_components
            self.n_iter = n_iter
            self.model = hmm.GaussianHMM(n_components=self.n_components, n_iter=self.n_iter)

        def train(self, X):
            self.model.fit(X)

    # Iterate through the data
    for label in set(y_words):
        # Get data for the current label
        label_data = np.vstack([X[i] for i, y in enumerate(y_words) if y == label])

        # Create an instance of the HMMTrainer
        hmm_trainer = HMMTrainer()

        # Train the HMM model
        hmm_trainer.train(label_data)

        # Append the trained model and label to hmm_models
        hmm_models.append((hmm_trainer.model, label))

        # Clear the trainer to free up memory
        hmm_trainer = None

    # Define the test files
    test_files = [
        '/content/drive/MyDrive/Colab Notebooks/hmm-speech-recognition-0.1/audio/pineapple/pineapple01.wav',
        '/content/drive/MyDrive/Colab Notebooks/hmm-speech-recognition-0.1/audio/orange/orange04.wav',
        '/content/drive/MyDrive/Colab Notebooks/hmm-speech-recognition-0.1/audio/kiwi/kiwi04.wav',
        '/content/drive/MyDrive/Colab Notebooks/hmm-speech-recognition-0.1/audio/apple/apple05.wav',
        '/content/drive/MyDrive/Colab Notebooks/hmm-speech-recognition-0.1/audio/peach/peach06.wav',
        '/content/drive/MyDrive/Colab Notebooks/hmm-speech-recognition-0.1/audio/lime/lime02.wav',
        '/content/drive/MyDrive/Colab Notebooks/hmm-speech-recognition-0.1/audio/banana/banana02.wav'
    ]

    # Classify input data using the trained models
    for input_file in test_files:
        # Read input file
        sampling_freq, audio = wavfile.read(input_file)
        # Extract MFCC features
        mfcc_features = mfcc(audio, sampling_freq)

        # Define variables to store the maximum score and the output label
        max_score = None
        output_label = None

        # Iterate through all HMM models and pick the one with the highest score
        for hmm_model, label in hmm_models:
            score = hmm_model.score(mfcc_features)
            if max_score is None or score > max_score:
                max_score = score
                output_label = label

        # Print the true and predicted labels
        true_label = input_file.split("/")[-1]  # Extract the true class label from the file path
        print("\nTrue:", true_label)
        print("Predicted:", output_label)
