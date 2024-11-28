import os
import pandas as pd
import wfdb
from tqdm import tqdm
from sklearn.model_selection import train_test_split

data_path = "/students/lifecare/team6/data"
metadata_path = os.path.join(data_path, "metadata.csv")
scp_statements_path = os.path.join(data_path, "scp_statements.csv")
output_path = [os.path.join(data_path, "processed_data/100Hz"), os.path.join(data_path, "processed_data/500Hz")]
for i in output_path:
    os.makedirs(i, exist_ok=True)
metadata = pd.read_csv(metadata_path)

def save_lead_I_files(metadata, output_path):
    for set_name in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(output_path, set_name), exist_ok=True)

    for set_name in ['train', 'validation', 'test']:
        set_folder = os.path.join(output_path, set_name)

        subset = metadata[metadata['set'] == set_name]
        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Processing {set_name} set"):
            if '100Hz' in output_path:
                hea_path = f"{row.file_path_lr}.hea"
                dat_path = f"{row.file_path_lr}.dat"
            else:
                hea_path = f"{row.file_path_hr}.hea"
                dat_path = f"{row.file_path_hr}.dat"

            try:
                record = wfdb.rdrecord(hea_path.replace(".hea", ""))

                lead_index = record.sig_name.index("I")
                lead_I_signal = record.p_signal[:, lead_index]

                record_name = os.path.splitext(os.path.basename(hea_path))[0]
                record_name = record_name.replace('.', '_')
                new_record_path = os.path.join(set_folder, record_name)

                wfdb.wrsamp(
                    record_name,
                    fs=record.fs,
                    units=[record.units[lead_index]],
                    sig_name=["I"],
                    p_signal=lead_I_signal.reshape(-1, 1),
                    write_dir=set_folder
                )

            except Exception as e:
                print(f"Error processing {hea_path}: {e}")

for i in output_path:
    save_lead_I_files(metadata, i)
    print(f"Lead I files processing and saving completed in {i}")
