import joblib

from src import preprocessing, config
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import os

def parse_args():
    p = argparse.ArgumentParser(description="train")
    # argument test size and random state
    p.add_argument("--random_state", "-r", type=int, default=config.random_state, help="random state")
    p.add_argument("--test_size", "-t", type=float, default=config.test_size, help="test size")
    # name model
    p.add_argument("--model_name","-m", type=str,default="LinearRegression",help="choies model")

    # argument randomForestRegression
    p.add_argument("--n_estimators","-n",type=int,default=300,help="number n_estimators")

    # argument Ridge
    p.add_argument("--alpha","-a", type=float ,default=1.0,help="number alpha")
    return p.parse_args()


def build_model(args):
    if args.model_name == "LinearRegression":
        clf = LinearRegression(
            fit_intercept=True,
            n_jobs=-1
        )
    elif args.model_name == "RandomForestRegressor":
        clf = RandomForestRegressor(
            n_estimators= args.n_estimators,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=args.random_state,
            n_jobs=-1
        )

    elif args.model_name == "Ridge":
        clf = Ridge(
            alpha=args.alpha,
            random_state=args.random_state
        )
    else:
        raise ValueError(
            f"Model '{args.model_name}' is not supported. "
            "Choose from: LinearRegression, RandomForestRegressor, Ridge"
        )
    return clf


def main(args):
    #  retrieve data
    x_train, x_test,y_train,y_test = preprocessing.preprocess_and_split(args.test_size,args.random_state)

    # create pipeline standardization
    # numerical
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    #nominal
    nom_transformer = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("one_hot",OneHotEncoder(handle_unknown="ignore"))
    ])
    #ordinal
    ord_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))
    ])

    # preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ("num_feature",num_transformer,config.numerical_cols),
        ("ord_feature",ord_transformer,config.ordinal_cols),
        ("nom_feature",nom_transformer,config.nominal_cols),
    ])

    # create pipline model
    clf = build_model(args)
    pipeline  = Pipeline(steps=[
        ("preprocessor",preprocessor),
        ("clf",clf)
    ])

    # train
    pipeline.fit(x_train,y_train)

    # test model
    y_predict = pipeline.predict(x_test)

    # model evaluation
    # the smaller the better
    print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
    print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
    # the bigger the better
    print("R2: {}".format(r2_score(y_test, y_predict)))

    if not os.path.isdir(config.reports_results_dir):
        os.makedirs(config.reports_results_dir)
    path_result = os.path.join(config.reports_results_dir, f"train_report_{args.model_name}.txt")
    with open(path_result,"w") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"MAE: {mean_absolute_error(y_test, y_predict):.4f}\n")
        f.write(f"MSE: {mean_squared_error(y_test, y_predict):.4f}\n")
        f.write(f"R2 : {r2_score(y_test, y_predict):.4f}\n")

    # save model
    if not os.path.isdir(config.dir_model):
        os.makedirs(config.dir_model)
    model_name = "{}.pkl".format(args.model_name)
    model_path = os.path.join(config.dir_model,model_name)
    joblib.dump(pipeline,model_path)
    print("save model successfull")


if __name__ == '__main__':
    args = parse_args()
    main(args)



