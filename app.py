from flask import Flask
from src import config
import os
from routes.predict_route import  predict_bp
from routes.report_route import report_bp


# tạo hai thư mục lưu file người dùng upload và kết quả dự đoán
upload = config.upload_folder
result = config.result_folder
os.makedirs(upload, exist_ok=True)
os.makedirs(result, exist_ok=True)

app= Flask(__name__)


app.register_blueprint(predict_bp)
app.register_blueprint(report_bp)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )