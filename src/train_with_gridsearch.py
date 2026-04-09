from src import preprocessing, config
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(description="GridSearch")

    p.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="Ridge"
    )

    p.add_argument(
        "--random_state",
        "-r",
        type=int,
        default=config.random_state
    )

    return p.parse_args()


def build_model_and_grid(args):
    if args.model_name == "LinearRegression":
        clf = LinearRegression(n_jobs=-1)
        param_grid = {
            "clf__fit_intercept": [True, False]
        }

    elif args.model_name == "RandomForestRegressor":
        clf = RandomForestRegressor(
            random_state=args.random_state,
            n_jobs=-1
        )
        param_grid = {
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [8, 12, 16],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 3],
        }

    elif args.model_name == "Ridge":
        clf = Ridge(random_state=args.random_state)
        param_grid = {
            "clf__alpha": [0.01, 0.1, 1.0, 10.0]
        }
    else:
        raise ValueError("Model not supported")

    return clf, param_grid


def main(args):
    x_train, x_test, y_train, y_test = preprocessing.preprocess_and_split()


    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    nom_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore"))
    ])

    ord_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num_feature", num_transformer, config.numerical_cols),
        ("ord_feature", ord_transformer, config.ordinal_cols),
        ("nom_feature", nom_transformer, config.nominal_cols),
    ])

    clf, param_grid = build_model_and_grid(args)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])


    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=2
    )

    grid.fit(x_train, y_train)

    print("BEST PARAMS:")
    print(grid.best_params_)
    print("BEST SCORE:", grid.best_score_)

    os.makedirs(config.dir_parameter, exist_ok=True)

    save_path = os.path.join(config.dir_parameter, f"best_params_{args.model_name}.txt")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_name}\n\n")
        f.write("Best Parameters:\n")
        for k, v in grid.best_params_.items():
            f.write(f"{k}: {v}\n")
        f.write("\nBest CV Score (R2):\n")
        f.write(f"{grid.best_score_:.6f}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)