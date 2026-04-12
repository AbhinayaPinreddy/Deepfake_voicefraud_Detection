import librosa
import numpy as np

def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

def trim_silence(audio):
    trimmed, _ = librosa.effects.trim(audio)
    return trimmed

def preprocess(file_path):
    audio = load_audio(file_path)
    audio = trim_silence(audio)
    audio = normalize_audio(audio)
    return audio