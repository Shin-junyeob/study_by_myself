import os
import pandas as pd
import numpy as np
import wfdb
import pickle
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

def load_dataset(metadata_path, data_path, Hz):
    metadata = pd.read_csv(metadata_path)

    def load_signals(set_name, Hz):
        subset = metadata[metadata['set'] == set_name]
        signals, labels = [], []
        for _, row in tqdm(subset.iterrows(), total=subset.shape[0], desc=f"Loading {set_name} signals"):
            if Hz == 100:
                file_path = os.path.join(data_path, f'{Hz}Hz', set_name, f"{row['file_name'].split('/')[1]}.dat")
            else:
                file_path = os.path.join(data_path, f'{Hz}Hz', set_name, f"{row['file_name'].split('/')[1].split('_')[0]}_hr.dat")
            try:
                signal, fields = wfdb.rdsamp(file_path.replace('.dat', ''), sampfrom=0, sampto=Hz*10)
                signals.append(signal.flatten())
                labels.append(row['sub_label'])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return np.array(signals, dtype=np.float32), labels

    X_train, y_train = load_signals('train', Hz)
    X_val, y_val = load_signals('validation', Hz)
    X_test, y_test = load_signals('test', Hz)

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

def smote_augment(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Original data shape: {X.shape, y.shape}")
    print(f"Resampled data shape: {X_res.shape, y_res.shape}")
    return X_res, y_res

def train_and_evaluate_binary(model_params, X_train, y_train, X_test, y_test, encoder, output_path, stage_name, threshold=0.4):
    print(f"Training {stage_name} ...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        params=model_params,
        dtrain=dtrain,
        num_boost_round=200
    )

    y_probs = model.predict(dtest)
    y_pred = (y_probs > threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)

    print(f"Stage: {stage_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    os.makedirs(output_path, exist_ok=True)
    results = {"accuracy": accuracy, "classification_report": report, "threshold": threshold}
    with open(os.path.join(output_path, f"fourth_{stage_name}_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved as fourth_{stage_name}_results.pkl")

    return accuracy, report

def train_and_evaluate_multiclass(model_params, X_train, y_train, X_test, y_test, encoder, output_path, stage_name):
    print(f"Training {stage_name} ...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        params=model_params,
        dtrain=dtrain,
        num_boost_round=200
    )

    y_pred = model.predict(dtest).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)

    print(f"Stage: {stage_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    os.makedirs(output_path, exist_ok=True)
    results = {"accuracy": accuracy, "classification_report": report}
    with open(os.path.join(output_path, f"fourth_{stage_name}_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved as fourth_{stage_name}_results.pkl")

    return accuracy, report

def main():
    metadata_path = '../data/metadata.csv'
    data_path = '../data/processed_data'
    Hz = 500
    output_path = f'../results/XGBoost/{Hz}Hz'

    (X_train, y_train), (_, _), (X_test, y_test) = load_dataset(metadata_path, data_path, Hz)

    binary_label_map = {"NORM": "NORM_MI", "MI": "NORM_MI", "STTC": "OTHERS", "HYP": "OTHERS", "CD": "OTHERS", "RHYTHM": "OTHERS", 'OTHER': 'OTHERS'}
    X_train_bin, y_train_bin, scaler, binary_encoder = preprocess_data(X_train, y_train, binary_label_map, fit_scaler=True)
    X_test_bin, y_test_bin, _, _ = preprocess_data(X_test, y_test, binary_label_map, scaler=scaler, encoder=binary_encoder)

    X_train_bin, y_train_bin = smote_augment(X_train_bin, y_train_bin)

    binary_model_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'learning_rate': 0.05,
        'max_depth': 3
    }
    train_and_evaluate_binary(binary_model_params, X_train_bin, y_train_bin, X_test_bin, y_test_bin, binary_encoder, output_path, "Binary_NORM_MI_vs_OTHERS", threshold=0.4)

    others_indices = [i for i, label in enumerate(y_train) if binary_label_map[label] == "OTHERS"]
    X_train_others = X_train[others_indices]
    y_train_others = [y_train[i] for i in others_indices]

    others_indices_test = [i for i, label in enumerate(y_test) if binary_label_map[label] == "OTHERS"]
    X_test_others = X_test[others_indices_test]
    y_test_others = [y_test[i] for i in others_indices_test]

    others_label_map = {label: label for label in ["STTC", "HYP", "CD", "RHYTHM"]}
    X_train_others, y_train_others, scaler_others, others_encoder = preprocess_data(X_train_others, y_train_others, others_label_map, fit_scaler=True)
    X_test_others, y_test_others, _, _ = preprocess_data(X_test_others, y_test_others, others_label_map, scaler=scaler_others, encoder=others_encoder)

    multi_class_model_params = {
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'num_class': len(others_encoder.classes_),
        'tree_method': 'hist',
        'device': 'cuda',
        'learning_rate': 0.05,
        'max_depth': 3
    }
    train_and_evaluate_multiclass(multi_class_model_params, X_train_others, y_train_others, X_test_others, y_test_others, others_encoder, output_path, "MultiClass_OTHERS")

if __name__ == "__main__":
    main()
