import os
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from model import CNNLSTMModel, TransformerECGModel, CNNTransformerECGModel, ResNet1DModel, BiLSTMModel

def load_dataset(metadata_path, data_path):
    metadata = pd.read_csv(metadata_path)

    def load_signals(set_name):
        subset = metadata[metadata['set'] == set_name]
        signals, labels = [], []
        for _, row in subset.iterrows():
            file_path = os.path.join(data_path, set_name, f"{row['file_name'].split('/')[1]}.dat")
            try:
                signal, fields = wfdb.rdsamp(file_path.replace('.dat', ''))
                signals.append(signal.flatten())
                labels.append(row['sub_label'])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return np.array(signals), labels

    X_train, y_train = load_signals('train')
    X_val, y_val = load_signals('validation')
    X_test, y_test = load_signals('test')
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def preprocess_data(X, y, fit_scaler=False, scaler=None, encoder=None):
    if scaler is None:
        scaler = StandardScaler()
    X = scaler.fit_transform(X) if fit_scaler else scaler.transform(X)

    if encoder is None:
        encoder = LabelEncoder()
    y = encoder.fit_transform(y) if fit_scaler else encoder.transform(y)

    return X, y, scaler, encoder

def train_and_evaluate_pytorch_model(model, X_train, y_train, X_test, y_test, encoder, output_path, model_name):
    print(f'Operating about {model_name} ...')
    X_train = torch.tensor(X_train[:, np.newaxis, :], dtype=torch.float32)
    X_test = torch.tensor(X_test[:, np.newaxis, :], dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False, num_workers=4)

    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(10):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch)
            y_pred.extend(torch.argmax(outputs, axis=1).numpy())

    accuracy = accuracy_score(y_test.numpy(), y_pred)
    report = classification_report(y_test.numpy(), y_pred, target_names=encoder.classes_)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    # Save results
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f"{model_name}_results.txt"), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(report)

    return accuracy, report

def main():
    metadata_path = '../data/metadata.csv'
    data_path = '../data/processed_data/100Hz'
    output_path = '../results/pytorch'

    # Load and preprocess data
    (X_train, y_train), (_, _), (X_test, y_test) = load_dataset(metadata_path, data_path)
    X_train, y_train, scaler, encoder = preprocess_data(X_train, y_train, fit_scaler=True)
    X_test, y_test, _, _ = preprocess_data(X_test, y_test, scaler=scaler, encoder=encoder)

    # Initialize models
    models = [
        {"model": CNNLSTMModel(input_dim=X_train.shape[1], hidden_dim=128, num_classes=len(encoder.classes_)), "name": "CNNLSTM"},
        {"model": TransformerECGModel(input_dim=X_train.shape[1], num_heads=4, hidden_dim=128, num_classes=len(encoder.classes_), num_layers=2), "name": "Transformer"},
        {"model": CNNTransformerECGModel(input_dim=X_train.shape[1], hidden_dim=128, num_heads=4, num_classes=len(encoder.classes_)), "name": "CNN+Transformer"},
        {"model": ResNet1DModel(input_dim=X_train.shape[1], num_classes=len(encoder.classes_)), "name": "ResNet1D"},
        {"model": BiLSTMModel(input_dim=X_train.shape[1], hidden_dim=128, num_classes=len(encoder.classes_)), "name": "Bi-LSTM"}
    ]
    
    results = []
    for entry in models:
        accuracy, report = train_and_evaluate_pytorch_model(entry["model"], X_train, y_train, X_test, y_test, encoder, output_path, entry["name"])
        results.append({"Model": entry["name"], "Accuracy": accuracy})

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_path, "results_summary.csv"), index=False)
    print(f"Results saved to {os.path.join(output_path, 'results_summary.csv')}")

    print("PyTorch analysis completed.")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()