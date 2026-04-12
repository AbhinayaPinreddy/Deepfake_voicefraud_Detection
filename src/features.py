import librosa
import numpy as np

def extract_features(audio, sr=16000):
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # Log-Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    log_mel = librosa.power_to_db(mel)

    # Combine
    features = np.vstack([mfcc, log_mel])
    
    return features