from flask import Blueprint,render_template,request,send_file
from services import validation_service,request_service,inference_service,file_service
from src import config
import os

upload = config.upload_folder
result = config.result_folder

predict_bp = Blueprint("Predict",__name__)


@predict_bp.route("/")
def home():
    return render_template("index.html",selected_model="LinearRegression",error=None,data=None,result=None,output_save=None)

@predict_bp.route("/predict",methods=["POST"])
def predict():
    try:
        # lấy ra tên model
        model_name = request_service.get_form_model(request.form)

        # lấy ra dư liệu người dùng nhập
        data = request_service.get_form_data(request.form)

        # validation
        validation_service.validate_dict(data)

        # dự đoán
        results = inference_service.predict_dict(data,model_name)

        return render_template("index.html", selected_model=model_name, error=None, data=data, result=results,
                               output_save=None)
    except Exception as e:

        model_name = request_service.get_form_model(request.form)
        return render_template("index.html", selected_model=model_name, error=str(e), data=request.form, result=None,
                               output_save=None)


@predict_bp.route("/predict_file",methods=["POST"])
def predict_file():
    try:
        # lấy ra model name
        model_name = request_service.get_form_model(request.form)

        # lấy file người dùng gửi
        file = request_service.get_form_file(request.files)

        # validation
        validation_service.validate_file(file)

        # save file upload
        input_path, name, timestamp = file_service.save_file_upload(file)

        # predict
        df_result = inference_service.predict_file(input_path,model_name)

        # save output
        output_filename = file_service.save_file_result(df_result,name,timestamp)

        return render_template("index.html", selected_model=model_name, error=None, data=None, result=None,
                               output_save=output_filename)
    except Exception as e:
        model_name = request_service.get_form_model(request.form)
        return render_template("index.html", selected_model=model_name, error=str(e), data=None, result=None,
                               output_save=None)

@predict_bp.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(result, filename)
    return send_file(
        file_path,
        as_attachment=True
    )
