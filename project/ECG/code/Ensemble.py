import os
import pickle
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CNNTransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_heads, num_layers, dropout):
        super(CNNTransformerModel, self).__init__()
        self.cnn = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.linear_input = nn.Linear(input_dim, hidden_dim)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))
        x = x.permute(0, 2, 1)
        x = self.transformer(x, x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ResNet1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ResNet1D, self).__init__()
        self.resnet = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.fc = nn.Linear(input_dim // 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.resnet(x)
        x = x.mean(dim=2)
        x = self.fc(x)
        return x


def load_dataset(metadata_path, data_path, Hz):
    metadata = pd.read_csv(metadata_path)

    def load_signals(set_name):
        subset = metadata[metadata['set'] == set_name]
        signals, labels = [], []
        for _, row in tqdm(subset.iterrows(), total=subset.shape[0], desc=f"Loading {set_name} signals"):
            if Hz == 100:
                file_path = os.path.join(data_path, f'{Hz}Hz', set_name, f"{row['file_name'].split('/')[1]}.dat")
            else:
                file_path = os.path.join(data_path, f'{Hz}Hz', set_name, f"{row['file_name'].split('/')[1].split('_')[0]}.dat")
            try:
                signal = wfdb.rdsamp(file_path.replace('.dat', ''), sampfrom=0, sampto=Hz * 10)[0]
                signals.append(signal.flatten())
                labels.append(row['sub_label'])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return np.array(signals, dtype=np.float32), labels

    X_train, y_train = load_signals('train')
    X_val, y_val = load_signals('validation')
    X_test, y_test = load_signals('test')

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def preprocess_data(X, y, label_map, fit_scaler=False, scaler=None, encoder=None):
    if scaler is None:
        scaler = StandardScaler()
    X = scaler.fit_transform(X) if fit_scaler else scaler.transform(X)

    if encoder is None:
        encoder = LabelEncoder()

    y_mapped = [label_map.get(label, "OTHERS") for label in y]
    y = encoder.fit_transform(y_mapped) if fit_scaler else encoder.transform(y_mapped)

    return X, y, scaler, encoder


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, axis=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {accuracy:.4f}")


def ensemble_predict(models, test_loader, device):
    all_preds = []
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        model_probs = [torch.softmax(model(X_batch), dim=1) for model in models]
        avg_probs = torch.stack(model_probs).mean(dim=0)
        preds = torch.argmax(avg_probs, axis=1)
        all_preds.extend(preds.cpu().numpy())
    return all_preds


def main():
    metadata_path = '../data/metadata.csv'
    data_path = '../data/processed_data'
    Hz = 100
    output_path = f'../results/Ensemble/{Hz}Hz'

    params = {
        100: {'hidden_dim': 128, 'num_heads': 4, 'num_layers': 3, 'dropout': 0.105, 'lr': 7.54e-5, 'batch_size': 32},
        500: {'hidden_dim': 256, 'num_heads': 8, 'num_layers': 3, 'dropout': 0.102, 'lr': 1.46e-4, 'batch_size': 32}
    }
    hyperparams = params[Hz]

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(metadata_path, data_path, Hz)
    label_map = {"NORM": 0, "MI": 1, "OTHERS": 2}
    X_train, y_train, scaler, encoder = preprocess_data(X_train, y_train, label_map, fit_scaler=True)
    X_val, y_val, _, _ = preprocess_data(X_val, y_val, label_map, scaler=scaler, encoder=encoder)
    X_test, y_test, _, _ = preprocess_data(X_test, y_test, label_map, scaler=scaler, encoder=encoder)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_transformer = CNNTransformerModel(
        input_dim=X_train.shape[1],
        num_classes=len(encoder.classes_),
        hidden_dim=hyperparams['hidden_dim'],
        num_heads=hyperparams['num_heads'],
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout']
    ).to(device)

    resnet1d = ResNet1D(input_dim=X_train.shape[1], num_classes=len(encoder.classes_)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_cnn_transformer = torch.optim.Adam(cnn_transformer.parameters(), lr=hyperparams['lr'])
    optimizer_resnet1d = torch.optim.Adam(resnet1d.parameters(), lr=hyperparams['lr'])

    print("\n--- Training CNN+Transformer ---")
    train_model(cnn_transformer, train_loader, val_loader, criterion, optimizer_cnn_transformer, device)

    print("\n--- Training ResNet1D ---")
    train_model(resnet1d, train_loader, val_loader, criterion, optimizer_resnet1d, device)

    print("\n--- Ensemble Evaluation ---")
    models = [cnn_transformer, resnet1d]
    all_preds = ensemble_predict(models, test_loader, device)

    test_accuracy = accuracy_score(y_test, all_preds)
    class_labels = ['NORM', 'MI', 'OTHERS']
    report = classification_report(y_test, all_preds, target_names=class_labels, output_dict=True)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "ensemble_results.pkl"), "wb") as f:
        pickle.dump({"accuracy": test_accuracy, "classification_report": report}, f)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Report:\n{classification_report(y_test, all_preds, target_names=class_labels)}")


if __name__ == "__main__":
    main()