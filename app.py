from flask import Flask,render_template, request,send_file
from validation import validation_input
from src import config,inference
import os
from datetime import datetime


# tạo hai thư mục lưu file người dùng upload và kết quả dự đoán
upload = config.upload_folder
result = config.result_folder
os.makedirs(upload, exist_ok=True)
os.makedirs(result, exist_ok=True)

app= Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html",selected_model="LinearRegression",error=None,data=None,result=None,output_save=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # lấy ra tên model
        model_name = request.form["model_name"]

        # lấy dữ liệu người dùng nhập vào
        data = {
            "Country": request.form["Country"],
            "Year": int(request.form["Year"]),
            "Status": request.form["Status"],
            "Adult Mortality": int(request.form["Adult_Mortality"]),
            "Alcohol": float(request.form["Alcohol"]),
            "Hepatitis B": float(request.form["Hepatitis_B"]),
            "Measles": int(request.form["Measles"]),
            "BMI": float(request.form["BMI"]),
            "under-five deaths": int(request.form["under_five_deaths"]),
            "Polio": float(request.form["Polio"]),
            "Total expenditure": float(request.form["Total_expenditure"]),
            "Diphtheria": float(request.form["Diphtheria"]),
            "HIV/AIDS": float(request.form["HIV_AIDS"]),
            "Income composition of resources": float(request.form["Income_composition_of_resources"]),
            "Schooling": int(request.form["Schooling"])
        }

        # validation
        validation_input.validate(data)

        #predict
        result = inference.model_predict_dic(data,model_name)
        print(result)

        return render_template("index.html", selected_model=model_name,error=None,data=data,result=result,output_save=None)

    except Exception as e:
        model_name = request.form.get("model_name","LinearRegression")
        return render_template("index.html", selected_model=model_name,error=str(e),data=request.form,result=None,output_save=None)


@app.route("/predict_file",methods=["POST"])
def predict_file():
    try:
        # lấy ra tên model
        model_name = request.form["model_name"]

        # lấy ra tên file upload
        file = request.files["file"]
        filename = file.filename

        # validation file name
        validation_input.validate_file(filename)

        # tạo tên file theo thời gian hiện tại
        name , ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
        new_filename = f"{name}_{timestamp}{ext}"

        # lưu file người dùng upload
        input_path = os.path.join(upload, new_filename)
        file.save(input_path)

        # tên file kết quả
        output_filename = f"{name}_prediction_{timestamp}.csv"

        # dư đoán
        df_result = inference.model_predict_file(input_path, model_name)

        # lưu file kết quả
        output_path = os.path.join(result, output_filename)
        df_result.to_csv(output_path, index=False)

        return render_template("index.html", selected_model=model_name, error=None, data=None, result=None,output_save=output_filename)
    except Exception as e:
        model_name = request.form.get("model_name", "LinearRegression")
        return render_template("index.html", selected_model=model_name, error=str(e), data=None, result=None,output_save=None)


@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(result, filename)
    return send_file(
        file_path,
        as_attachment=True
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )