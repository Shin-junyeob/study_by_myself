import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from model import RandomForestModel, CNNLSTMModel
import torch
import torch.nn as nn
import torch.optim as optim
import wfdb


def load_dataset(metadata_path, data_path):
    """
    Load dataset from metadata and signal files.
    """
    metadata = pd.read_csv(metadata_path)

    def load_signals(set_name):
        subset = metadata[metadata['set'] == set_name]
        signals = []; labels = []
        for _, row in subset.iterrows():
            base_name = os.path.basename(row['filename_lr'])
            dat_path = os.path.join(data_path, set_name, base_name + '.dat')
            hea_path = os.path.join(data_path, set_name, base_name + '.hea')
            try:
                record = wfdb.rdsamp(dat_path.replace('.dat', ''))
                signals.append(record[0].flatten())
                labels.append(row['sub_label'])
            except Exception as e:
                print(f"Error loading {dat_path}: {e}")
        return np.array(signals), labels

    X_train, y_train = load_signals('train')
    X_val, y_val = load_signals('validation')
    X_test, y_test = load_signals('test')

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def preprocess_data(X, y, fit_scaler=False, scaler=None, encoder=None):
    """
    Standardize X and encode y.
    """
    if scaler is None:
        scaler = StandardScaler()

    X = scaler.fit_transform(X) if fit_scaler else scaler.transform(X)

    if encoder is None:
        encoder = LabelEncoder()

    if fit_scaler:
        y = encoder.fit_transform(y)
    else:
        y = encoder.transform(y)

    return X, y, scaler, encoder


def train_cnn_lstm(model, train_loader, val_loader, num_epochs, device):
    """
    Train CNN-LSTM model using PyTorch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_accuracy = correct / total
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    return model


def main():
    metadata_path = '../data/metadata.csv'
    data_path = '../data/processed_data/100Hz'
    output_path = '../results'
    os.makedirs(output_path, exist_ok=True)

    # Load and preprocess data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(metadata_path, data_path)
    X_train, y_train, scaler, encoder = preprocess_data(X_train, y_train, fit_scaler=True)
    X_val, y_val, _, _ = preprocess_data(X_val, y_val, scaler=scaler, encoder=encoder)
    X_test, y_test, _, _ = preprocess_data(X_test, y_test, scaler=scaler, encoder=encoder)

    # Initialize models
    models = [
        RandomForestModel(n_estimators=100),
        CNNLSTMModel(input_dim=X_train.shape[1], hidden_dim=128, num_classes=len(encoder.classes_))
    ]

    # Train and evaluate models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models:
        model_name = model.__class__.__name__

        if model_name == 'RandomForestModel':
            print(f"Training {model_name}...")
            model.fit(X_train, y_train)
            print(f"Evaluating {model_name}...")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=encoder.classes_)
        else:
            print(f"Training {model_name}...")
            X_train_cnn = torch.tensor(X_train[:, :, np.newaxis], dtype=torch.float32)
            X_val_cnn = torch.tensor(X_val[:, :, np.newaxis], dtype=torch.float32)
            X_test_cnn = torch.tensor(X_test[:, :, np.newaxis], dtype=torch.float32)
            y_train_torch = torch.tensor(y_train, dtype=torch.long)
            y_val_torch = torch.tensor(y_val, dtype=torch.long)
            y_test_torch = torch.tensor(y_test, dtype=torch.long)

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            model.fit(X_train_cnn, y_train_torch, X_val_cnn, y_val_torch, optimizer, criterion, epochs=10)

            print(f"Evaluating {model_name}...")
            y_pred = model.predict(X_test_cnn)
            y_pred_classes = np.argmax(y_pred, axis=1)
            accuracy = accuracy_score(y_test, y_pred_classes)
            report = classification_report(y_test, y_pred_classes, target_names=encoder.classes_)

        print(f"Accuracy for {model_name}: {accuracy:.4f}")
        print(f"Classification Report for {model_name}:\n{report}")

        # Save results
        with open(os.path.join(output_path, f"{model_name}_results.txt"), 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(report)


if __name__ == "__main__":
    main()
