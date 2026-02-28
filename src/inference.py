import os
from src import config, preprocessing
import joblib
import pandas as pd


def load_model(model_name):
    model_path = os.path.join(config.dir_model,"{}.pkl".format(model_name))
    if not os.path.isfile(model_path):
        raise FileExistsError(f"Model not found: {model_path}")
    return joblib.load(model_path)

def model_predict_dic(input_dic,model_name):
    #load model
    model = load_model(model_name)

    # convert dic to dataframe
    df = pd.DataFrame([input_dic])

    # clear data
    df = preprocessing.clean_raw_data(df,False)

    # predict
    prediction = round(float(model.predict(df)[0]), 2)

    return prediction

def model_predict_file(input_file,model_name):
    # load model
    model = load_model(model_name)

    # load data
    if input_file.endswith(".csv"):
        try:
            df = pd.read_csv(input_file)
        except Exception:
            raise ValueError("can not load file this csv ")
    elif input_file.endswith(".xlsx") or input_file.endswith(".xls"):
        try:
            df = pd.read_excel(input_file)
        except Exception:
            raise ValueError("can not load file this exel ")
    else:
        raise ValueError("Only CSV or Excel files are supported")

    # clear data
    df = preprocessing.clean_raw_data(df,False)

    # prediction
    prediction = model.predict(df)

    df["prediction"] = prediction

    return df

if __name__ == '__main__':
    # dic
    input_dic = {
        "Country": "Vietnam",
        "Year": 2015,
        "Status": "Developing",
        "Adult Mortality": 142,
        "infant deaths": 18,
        "Alcohol": 4.3,
        "percentage expenditure": 0.0,
        "Hepatitis B": 97,
        "Measles": 0,
        "BMI": 23.5,
        "under-five deaths": 22,
        "Polio": 96,
        "Total expenditure": 5.8,
        "Diphtheria": 96,
        "HIV/AIDS": 0.1,
        "GDP": 2600,
        "Population": 91700000,
        "thinness  1-19 years": 12.3,
        "thinness 5-9 years": 13.0,
        "Income composition of resources": 0.68,
        "Schooling": 12.5
    }
    print(model_predict_dic(input_dic,"Ridge"))

    # file
    test_file = os.path.join(config.data_processed,"x_test.csv")
    df_result = model_predict_file(test_file,"Ridge")
    print(df_result.head())