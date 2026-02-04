from src import config
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_data():
    return pd.read_csv(config.data_raw_path)

def clean_raw_data(df,is_train=True):
    # delete raw duplication
    df = df.drop_duplicates()

    # remove white space
    df.columns = df.columns.str.strip()

    # remove unnecessary columns
    df = df.drop(config.unnecessary_cols, axis=1, errors="ignore")

    # keep the necessary
    if is_train:
        necessary_cols = (
            config.numerical_cols +
            config.category_cols +
            config.ordinal_cols +
            [config.target_col]
        )
        df = df[necessary_cols]
    else:
        necessary_cols = (
            config.numerical_cols +
            config.category_cols +
            config.ordinal_cols
        )
        df = df[necessary_cols]

    return df

def preprocess_and_split(test_size=None,random_state=None):
    # root data processed
    processed_dir = config.data_processed

    if test_size is None:
        test_size = config.test_size
    if random_state is None:
        random_state = config.random_state

    # load data
    df = load_data()

    # clear data
    df = clean_raw_data(df,True)

    # split target, sample
    x = df.drop(config.target_col,axis=1)
    y = df[config.target_col]

    # train , test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)

    x_train.to_csv(os.path.join(processed_dir, "x_train.csv"), index=False)
    x_test.to_csv(os.path.join(processed_dir, "x_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = preprocess_and_split()
    print(x_train.head(2))
    print(x_test.head(2))
    print(y_train.head(2))
    print(y_test.head(2))
