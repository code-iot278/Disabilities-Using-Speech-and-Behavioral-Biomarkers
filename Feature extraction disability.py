import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# =====================================================
# 1. Conv-BiLSTM Feature Extractor
# =====================================================
class ConvBiLSTM(nn.Module):
    def __init__(self, input_channels, conv_filters, lstm_hidden):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, conv_filters, 3, padding=1),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(conv_filters, conv_filters * 2, 3, padding=1),
            nn.BatchNorm1d(conv_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.bilstm = nn.LSTM(
            input_size=conv_filters * 2,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.bilstm(x)
        return torch.mean(out, dim=1)


# =====================================================
# 2. Graph Attention Network
# =====================================================
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        Wh = self.W(h)
        N = Wh.size(0)

        a_input = torch.cat([
            Wh.repeat(1, N).view(N * N, -1),
            Wh.repeat(N, 1)
        ], dim=1).view(N, N, -1)

        e = self.leakyrelu(self.a(a_input).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        return torch.matmul(attention, Wh)


class GAT(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.gat1 = GATLayer(in_features, hidden)
        self.gat2 = GATLayer(hidden, out_features)

    def forward(self, x, adj):
        x = F.elu(self.gat1(x, adj))
        return self.gat2(x, adj)


# =====================================================
# 3. Feature Extraction Model
# =====================================================
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.conv_bilstm = ConvBiLSTM(
            input_channels=input_channels,
            conv_filters=64,
            lstm_hidden=128
        )

        self.gat = GAT(
            in_features=256,
            hidden=128,
            out_features=128
        )

    def forward(self, x, adj):
        features = self.conv_bilstm(x)
        return self.gat(features, adj)


# =====================================================
# 4. CSV → Feature Extraction → CSV
# =====================================================
def extract_features_from_csv(input_csv, output_csv):

    # Load CSV
    df = pd.read_csv(input_csv)
    data = df.values.astype(np.float32)

    # Convert to tensor
    # [samples, channels=1, time_steps=features]
    x = torch.tensor(data).unsqueeze(1)

    # Build adjacency (fully connected graph)
    num_samples = x.size(0)
    adj = torch.ones(num_samples, num_samples)

    # Model
    model = FeatureExtractor(input_channels=1)
    model.eval()

    with torch.no_grad():
        features = model(x, adj)

    # Save to CSV
    feature_df = pd.DataFrame(features.numpy())
    feature_df.to_csv(output_csv, index=False)

    print("Input CSV  :", input_csv)
    print("Output CSV :", output_csv)
    print("Feature shape:", features.shape)


# =====================================================
# 5. Run
# =====================================================
if __name__ == "__main__":

    INPUT_CSV = ""      # your input file
    OUTPUT_CSV = ""

    extract_features_from_csv(INPUT_CSV, OUTPUT_CSV)
