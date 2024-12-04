import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


class TransformerECGModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_classes, num_layers):
        super(TransformerECGModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)  # Adjust input_dim to match the data
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x.mean(dim=1))
        return x


class CNNTransformerECGModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_classes):
        super(CNNTransformerECGModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=2
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Expected input: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # Change shape to (batch_size, input_dim, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.transpose(1, 2)  # Back to (batch_size, seq_len, input_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average over sequence
        x = self.fc(x)
        return x


class ResNet1DModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ResNet1DModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        for _ in range(blocks):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            stride = 1  # Only first block applies stride
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        return x


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Last time step output
        return x