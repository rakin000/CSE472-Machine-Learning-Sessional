import pandas as pd
import opendatasets  as od
from data_preprocessor_helpers import *

def preprocess(k=50):
    od.download('https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data')
    data = pd.read_csv('creditcardfraud/creditcard.csv')

    binary_category_features = ['Class'] #,
    numerical_features = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12', 'V13','V14','V15','V16','V17','V18','V19','V20','V21','V22', 'V23', 'V24','V25','V26','V27','V28','Amount']

    from sklearn.utils import resample

    data_majority = data[data['Class'] == 0]
    data_minority = data[data['Class'] == 1]

    # print(f"{len(data_majority)=}")
    # print(f"{len(data_minority)=}")

    data_majority_sampled = resample(data_majority, replace=True, n_samples=10000, random_state=5)
    data_minority_sampled = resample(data_minority, replace=True, n_samples=10000, random_state=5)

    data_balanced = pd.concat([data_majority_sampled, data_minority_sampled])

    # data_balanced["Class"].value_counts().plot(kind="pie",autopct='%1.0f%%')

    from sklearn.model_selection import train_test_split

    y = data_balanced['Class']
    X = data_balanced.drop('Class',axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)

    normalizers = normalizer_(X_train, numerical_features)
    for feature in numerical_features:
        X_test[feature] = normalizers[feature].transform(X_test[[feature]])


    information_gain_feature = dict()
    for feature in X_train.columns:
        information_gain_feature[feature] = calculate_information_gain(pd.concat([X_train, y_train], axis=1), feature, 'Class')

    sorted_dict_asc = dict(sorted(information_gain_feature.items(), key=lambda item: item[1], reverse=True))
    sorted_dict_asc

    list(sorted_dict_asc.keys())

    top_k_features = list(sorted_dict_asc.keys())[:k]

    X_train_top_k = X_train[top_k_features]
    X_test_top_k = X_test[top_k_features]

    return X_train_top_k, y_train, X_test_top_k, y_test