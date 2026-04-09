from lazypredict.Supervised import LazyRegressor
from src import preprocessing,config

x_train, x_test, y_train, y_test = preprocessing.preprocess_and_split()

clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)