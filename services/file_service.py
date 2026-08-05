from src import config
import os
from datetime import datetime


uploads = config.upload_folder
results = config.result_folder

def save_file_upload(file):
    filename = file.filename

    name,ext =  os.path.splitext(filename)
    timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    new_filename = f"{name}_{timestamp}{ext}"

    # lưu file người dùng upload
    input_path = os.path.join(uploads, new_filename)
    file.save(input_path)

    return input_path,name,timestamp


def save_file_result(df_result,name,timestamp):
    # tên file kết quả
    output_filename = f"{name}_prediction_{timestamp}.csv"

    # lưu file kết quả
    output_path = os.path.join(results, output_filename)
    df_result.to_csv(output_path, index=False)

    return output_filename