def get_form_data(form):
    data = {
        "Country": form["Country"],
        "Year": int(form["Year"]),
        "Status": form["Status"],
        "Adult Mortality": int(form["Adult_Mortality"]),
        "Alcohol": float(form["Alcohol"]),
        "Hepatitis B": float(form["Hepatitis_B"]),
        "Measles": int(form["Measles"]),
        "BMI": float(form["BMI"]),
        "under-five deaths": int(form["under_five_deaths"]),
        "Polio": float(form["Polio"]),
        "Total expenditure": float(form["Total_expenditure"]),
        "Diphtheria": float(form["Diphtheria"]),
        "HIV/AIDS": float(form["HIV_AIDS"]),
        "Income composition of resources": float(form["Income_composition_of_resources"]),
        "Schooling": int(form["Schooling"])
    }

    return data


def get_form_model(form):
    model_name = form.get("model_name", "logistic")
    return model_name


def get_form_file(files):
    file = files["file"]
    return file