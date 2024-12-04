import os
import pandas as pd
from sklearn.model_selection import train_test_split

data_path = "../data"
metadata_path = os.path.join(data_path, "ptbxl_database.csv")
scp_statements_path = os.path.join(data_path, "scp_statements.csv")
output_path = os.path.join(data_path, "processed_data")
os.makedirs(output_path, exist_ok=True)

metadata = pd.read_csv(metadata_path)
scp_statements = pd.read_csv(scp_statements_path, index_col=0)

diagnostic_classes = scp_statements['diagnostic_class'].dropna().unique()
diagnostic_mapping = scp_statements['diagnostic_class'].dropna().to_dict()

rhythm_classes = scp_statements[scp_statements['rhythm'] == 1].index.tolist()

metadata['file_path_lr'] = metadata['filename_lr'].apply(lambda x: os.path.join(data_path, x.split('.')[0]))
metadata['file_path_hr'] = metadata['filename_hr'].apply(lambda x: os.path.join(data_path, x.split('.')[0]))
metadata['file_name'] = metadata['filename_lr'].apply(lambda x: x.split('records100/')[1])

metadata['scp_codes'] = metadata['scp_codes'].apply(eval)

metadata['filtered_scp_codes'] = metadata['scp_codes'].apply(
    lambda codes: {code: value for code, value in codes.items() if code in diagnostic_mapping or code in rhythm_classes}
)
metadata['diagnostic_class'] = metadata['filtered_scp_codes'].apply(
    lambda codes: list({diagnostic_mapping[code] for code in codes.keys() if code in diagnostic_mapping})
)
metadata['rhythm_class'] = metadata['filtered_scp_codes'].apply(
    lambda codes: [code for code in codes.keys() if code in rhythm_classes]
)
# metadata['is_relevant'] = metadata['diagnostic_class'].apply(lambda classes: len(classes) > 0) | metadata['rhythm_class'].apply(lambda classes: len(classes) > 0)
# metadata = metadata[metadata['is_relevant']]

metadata['labels'] = metadata.apply(
    lambda row: ",".join(row['diagnostic_class'] + row['rhythm_class']), axis = 1
)

def classify_abnormal(labels):
    if 'NORM' in labels:
        return 'NORM'
    elif 'MI' in labels:
        return 'MI'
    elif 'STTC' in labels:
        return 'STTC'
    elif 'CD' in labels:
        return 'CD'
    elif 'HYP' in labels:
        return 'HYP'
    elif any(rhythm in labels for rhythm in ['AFIB', 'STACH', 'SBRAD', 'SVTAC', 'PSVT', 'AFLT', 'SVARR', 'TRIGU']):
        return 'RHYTHM'
    else:
        'OTHER'

metadata['sub_label'] = metadata['labels'].apply(classify_abnormal)

subject_ids = metadata['patient_id'].unique()
train_ids, test_ids = train_test_split(subject_ids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

metadata['set'] = metadata['patient_id'].apply(
    lambda x: 'train' if x in train_ids else 'validation' if x in val_ids else 'test'
)

metadata_output_path = os.path.join(data_path, "metadata.csv")
metadata.to_csv(metadata_output_path, index=False)
print(f"Updated metadata saved to {metadata_output_path}")