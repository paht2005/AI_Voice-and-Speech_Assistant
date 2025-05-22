# Copyright @[Phat Nguyen Cong) (https://github.com/paht2005)

# ✅ Feature 5: Emotion Detection
# 🎯 Objective: Analyze a speaker's voice to detect emotional states such as happy, sad, angry, or neutral.


import librosa
import numpy as np
import joblib

# Feature Extraction – MFCCs
def extract_mfcc(path):
    y, sr = librosa.load(path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Predict Emotion for New Audio
def predict_emotion(audio_path, model_path="emotion_model.pkl"): # audio in .wav format file
    model = joblib.load(model_path)
    feature = extract_mfcc(audio_path).reshape(1, -1)
    return model.predict(feature)[0]

"""
1) Key Concepts
    - MFCCs: Capture the timbre and tone from audio—key to detecting emotions
    - RandomForest or CNN can be used as classifiers
    - Dataset filenames often encode emotion info (e.g., 03 = happy in RAVDESS)
2) Feature Summary
    - Input: Voice file (WAV)
    - Output: Emotion label ( happy , angry , neutral , etc.)
    - Model: RandomForest trained on MFCCs (or use CNN/LSTM later)
    - Use Case: Sentiment AI, wellness monitoring, affective agents
"""







"""
✍️ Technical Description
This module performs emotion classification from speech. It works by extracting acoustic features from the user's voice and feeding them into a trained machine learning model.

Step-by-Step Workflow:
    1) Audio Feature Extraction
        - Use MFCC (Mel-Frequency Cepstral Coefficients), a widely used feature that captures tone, pitch, and timbre, which are indicative of emotional state.
        - Audio is typically trimmed or padded to a standard duration (e.g., 3 seconds) to normalize input length.

    2) Classification Model
        - The extracted MFCC vectors are input into a RandomForest classifier or a simple neural network trained on emotional speech datasets like RAVDESS, TESS, or CREMA-D.
        - The model predicts one of several predefined emotion categories (e.g., happy, sad, angry, neutral, etc.).
"""