import os
import pickle

# 변환할 pkl 파일들의 디렉토리 (현재 경로)
directory = './'

# .pkl 파일을 .txt 파일로 변환하는 함수
def convert_pkl_to_txt(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(directory, file_name)
            txt_file_path = os.path.join(directory, file_name.replace('.pkl', '.txt'))
            try:
                # pkl 파일 읽기
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # .txt 파일로 저장
                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                    # 데이터가 dictionary 형식인 경우
                    if isinstance(data, dict):
                        for key, value in data.items():
                            txt_file.write(f"=== {key.upper()} ===\n")
                            if isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    txt_file.write(f"{sub_key}: {sub_value}\n")
                            else:
                                txt_file.write(f"{value}\n")
                            txt_file.write("\n")
                    # 데이터가 DataFrame 형식인 경우
                    elif hasattr(data, 'to_string'):
                        txt_file.write(data.to_string())
                    # 기타 형식
                    else:
                        txt_file.write(str(data))

                print(f"Converted {file_name} to {txt_file_path}")

            except Exception as e:
                print(f"Error converting {file_name}: {e}")

# 실행
convert_pkl_to_txt(directory)
