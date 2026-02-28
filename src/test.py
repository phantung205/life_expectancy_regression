import os
from src import config,preprocessing
import argparse
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def parse_args():
    p = argparse.ArgumentParser(description="test model")
    p.add_argument("--model_name","-m",type=str,default="LinearRegression")

    return p.parse_args()
def main(args):
    # data
    _,x_test,_,y_test = preprocessing.preprocess_and_split()

    # load model
    model_path = os.path.join(config.dir_model,"{}.pkl".format(args.model_name))
    if not os.path.join(model_path):
        print("You need to train the model to have checkpoints before testing.")
        exit(0)
    model = joblib.load(model_path)

    y_pred = model.predict(x_test)

    # the smaller the better
    print("MAE: {}".format(mean_absolute_error(y_test, y_pred)))
    print("MSE: {}".format(mean_squared_error(y_test, y_pred)))
    # the bigger the better
    print("R2: {}".format(r2_score(y_test, y_pred)))

if __name__ == '__main__':
    args = parse_args()
    main(args)