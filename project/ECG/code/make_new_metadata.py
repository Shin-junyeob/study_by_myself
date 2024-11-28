import os
import pandas as pd
from sklearn.model_selection import train_test_split

data_path = "/students/lifecare/team6/data"
metadata_path = os.path.join(data_path, "ptbxl_database.csv")
scp_statements_path = os.path.join(data_path, "scp_statements.csv")
output_path = os.path.join(data_path, "processed_data")
os.makedirs(output_path, exist_ok=True)

metadata = pd.read_csv(metadata_path)
scp_statements = pd.read_csv(scp_statements_path, index_col=0)

diagnostic_classes = scp_statements['diagnostic_class'].dropna().unique()
diagnostic_mapping = scp_statements['diagnostic_class'].dropna().to_dict()

metadata['file_path_lr'] = metadata['filename_lr'].apply(lambda x: os.path.join(data_path, x.split('.')[0]))
metadata['file_path_hr'] = metadata['filename_hr'].apply(lambda x: os.path.join(data_path, x.split('.')[0]))
metadata['scp_codes'] = metadata['scp_codes'].apply(eval)

metadata['filtered_scp_codes'] = metadata['scp_codes'].apply(
    lambda codes: {code: value for code, value in codes.items() if code in diagnostic_mapping}
)
metadata['diagnostic_class'] = metadata['filtered_scp_codes'].apply(
    lambda codes: list({diagnostic_mapping[code] for code in codes.keys()})
)
metadata['is_diagnostic'] = metadata['diagnostic_class'].apply(lambda classes: len(classes) > 0)
metadata['labels'] = metadata['diagnostic_class'].apply(lambda classes: ",".join(classes))

metadata['binary_label'] = metadata['labels'].apply(
    lambda x: 0 if 'NORM' in x else 1
)

metadata = metadata[metadata['is_diagnostic']]

subject_ids = metadata['patient_id'].unique()
train_ids, test_ids = train_test_split(subject_ids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

metadata['set'] = metadata['patient_id'].apply(
    lambda x: 'train' if x in train_ids else 'validation' if x in val_ids else 'test'
)

metadata_output_path = os.path.join(data_path, "metadata.csv")
metadata.to_csv(metadata_output_path, index=False)
print(f"Updated metadata saved to {metadata_output_path}")