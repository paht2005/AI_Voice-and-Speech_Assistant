# Copyright @[Phat Nguyen Cong) (https://github.com/paht2005)
import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Emotion mapping from RAVDESS filename format
def get_emotion_label(filename):
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    return emotion_map.get(filename.split("-")[2], "unknown")

# Extract MFCC feature from audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Load dataset
X, y = [], []
data_dir = "ravdess/"

for file in os.listdir(data_dir):
    if file.endswith(".wav"):
        path = os.path.join(data_dir, file)
        label = get_emotion_label(file)
        features = extract_features(path)
        X.append(features)
        y.append(label)

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluation
print("\nClassification Report:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "emotion_model.pkl")
print("✅ Model saved as emotion_model.pkl")
