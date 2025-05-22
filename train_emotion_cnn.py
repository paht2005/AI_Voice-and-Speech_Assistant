# Copyright @[Phat Nguyen Cong) (https://github.com/paht2005)

import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load and extract features (MFCC -> image-like 2D)
def extract_mfcc_2d(file_path, max_pad_len=174):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

# Get emotion label from filename
def get_label(filename):
    return {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }.get(filename.split("-")[2], 'unknown')

# Load dataset
X, y = [], []
for file in os.listdir("ravdess-data"):
    if file.endswith(".wav"):
        mfcc = extract_mfcc_2d(os.path.join("ravdess-data", file))
        label = get_label(file)
        X.append(mfcc)
        y.append(label)

X = np.array(X)
X = X[:, np.newaxis, :, :]  # shape: (N, 1, 40, 174)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

# Save label encoder
joblib.dump(le, "emotion_label_encoder.pkl")

# Prepare tensors
tensor_x = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(tensor_x, y_tensor)

# Train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# CNN Model
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 40, 174)
            dummy_output = self.conv(dummy_input)
            self.flatten_dim = dummy_output.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = EmotionCNN(num_classes=len(le.classes_))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train loop
for epoch in range(15):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "emotion_cnn.pth")
print("✅ Saved CNN model as emotion_cnn.pth")
