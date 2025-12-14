import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# Input CSV File
# =========================
input_csv = "selected_features.csv"   # Output of HAS-ReliefF

# =========================
# Load Dataset
# =========================
df = pd.read_csv(input_csv)

# Assume last column is class label
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# =========================
# Train-Test Split (80/20)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

# =========================
# DataLoader
# =========================
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

# =========================
# Residual Block
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return self.relu(out + residual)

# =========================
# IndRNN Layer
# =========================
class IndRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

# =========================
# DeepSTNet Architecture
# =========================
class DeepSTNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.indrnn = IndRNN(input_dim, 128)
        self.res_block1 = ResidualBlock(128)
        self.res_block2 = ResidualBlock(128)

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.indrnn(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.classifier(x)

# =========================
# Model Initialization
# =========================
num_classes = len(np.unique(y))
model = DeepSTNet(input_dim=X_train.shape[1], num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# Model Training
# =========================
EPOCHS = 30

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f}")

# =========================
# Evaluation (Accuracy)
# =========================
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predictions = torch.argmax(outputs, dim=1)

accuracy = accuracy_score(y_test.numpy(), predictions.numpy())

print("\n==============================")
print(f"Disorder Detection Accuracy: {accuracy * 100:.2f}%")
print("==============================")
