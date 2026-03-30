import librosa
import numpy as np

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=22050)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )

    mel = np.mean(mel.T, axis=0)

    # Feature fusion
    features = np.concatenate((mfcc, mel))

    return features