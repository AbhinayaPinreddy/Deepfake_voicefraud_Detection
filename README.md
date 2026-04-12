# Deepfake Fraud Detection

## Description

This project implements a deepfake audio detection system using machine learning. It utilizes the ASVspoof2019 dataset to train a convolutional neural network (CNN) model for distinguishing between genuine and spoofed audio files.

## Features

- Audio preprocessing and feature extraction
- CNN-based model for binary classification
- Training and validation on ASVspoof2019 LA subset
- Prediction API for uploaded audio files

## Installation

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the training script:

```bash
python src/train.py
```

This will train the model using the ASVspoof2019 LA training and development sets.

### Making Predictions

To predict on a single audio file:

```bash
python src/predict.py
```

### Running the API

Start the FastAPI server:

```bash
uvicorn app.api:app --reload
```

Upload an audio file to /analyze-audio to get a prediction.

## Dataset

The project uses the ASVspoof2019 dataset, specifically the Logical Access (LA) subset. Since the data is large and not included in the repository (it's in .gitignore), you need to download it separately.

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset).

After downloading and extracting, place the contents in the `data/` directory to match the expected structure:

- Evaluation set: ASVspoof2019_LA_eval

## Model

The model is a CNN with convolutional layers, max pooling, and fully connected layers. It outputs a probability score for spoof detection.

Trained model is saved as models/model.pth
