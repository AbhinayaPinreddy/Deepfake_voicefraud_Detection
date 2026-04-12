import torch
from model import CNNModel
from preprocess import preprocess
from features import extract_features
import numpy as np

def pad_features(features, max_len=300):
    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)))
    else:
        features = features[:, :max_len]
    return features


def predict(file_path):
    model = CNNModel()
    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()

    audio = preprocess(file_path)
    features = extract_features(audio)
    features = pad_features(features)

    features = torch.tensor(features).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        output = model(features)
        prob = torch.sigmoid(output).item()

    print("Probability:", prob)  

    return {
        "prediction": "Synthetic" if prob > 0.6 else "Real",  
        "confidence": prob
    }