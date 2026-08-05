import os


def validate_dict(data):
    if data["Year"] <= 0:
        raise ValueError("Year phải lớn hơn  0")
    if data["Adult Mortality"] < 0:
        raise ValueError("Adult_Mortality phải lớn hơn hoặc bằng 0")
    if data["Alcohol"] < 0:
        raise ValueError("Alcohol phải lớn hơn hoặc bằng 0")
    if data["Hepatitis B"] < 0:
        raise ValueError("Hepatitis_B phải lớn hoặc bằng hơn 0")
    if data["Measles"] < 0:
        raise ValueError("Measles phải lớn hơn hoặc bằng 0")
    if data["BMI"] <= 0:
        raise ValueError("BMI phải lớn hơn 0")
    if data["under-five deaths"] < 0:
        raise ValueError("under_five_deaths phải lớn hơn hoặc bằng 0")
    if data["Polio"] < 0:
        raise ValueError("Polio phải lớn hơn hoặc bằng 0")
    if data["Total expenditure"] < 0:
        raise ValueError("Total_expenditure phải lớn hơn hoặc bằng 0")
    if data["Diphtheria"] < 0:
        raise ValueError("Diphtheria phải lớn hơn hoặc bằng 0")
    if data["HIV/AIDS"] < 0:
        raise ValueError("HIV_AIDS phải lớn hơn hoặc bằng 0")
    if data["Income composition of resources"] < 0:
        raise ValueError("Income_composition_of_resources phải lớn hơn hoặc bằng 0")
    if data["Schooling"] <= 0:
        raise ValueError("Schooling phải lớn hơn 0")

def validate_file(file):
    if file is None or file.filename == "":
        raise ValueError("Chưa chọn file")

    ext = os.path.splitext(file.filename)[1].lower()

    allowed_ext = {".csv", ".xlsx", ".xls"}

    if ext not in allowed_ext:
        raise ValueError("Chỉ hỗ trợ file CSV hoặc Excel")
