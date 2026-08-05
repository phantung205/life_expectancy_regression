from deploys import inference


def predict_dict(data,model_name):
    prediction = inference.model_predict_dic(data,model_name)
    return prediction


def predict_file(input_path,model_name):
    df_result = inference.model_predict_file(input_path,model_name)
    return df_result