import pandas as pd
import opendatasets  as od
from data_preprocessor_helpers import * 

def preprocess(k=70):
    data = pd.read_csv('https://archive.ics.uci.edu/static/public/2/data.csv')

    # data = data.dropna(subset=['workclass','occupation','native-country'])
    data = data.dropna().reset_index(drop=True)
    data.info()

    data['income'] = data['income'].replace('<=50K.','<=50K')
    data['income'] = data['income'].replace('>50K.','>50K')
    multi_category_features = ['workclass', 'education', 'marital-status',
                        'occupation', 'relationship', 'race', 'native-country',]
    binary_category_features = ['sex','income',] #,
    numerical_features = ['age', 'education-num', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week' ]
    category_features = binary_category_features+multi_category_features


    label_imapping = label_encoder_(data, binary_category_features)


    data = onehot_encoder_(data, multi_category_features)

    from sklearn.utils import resample

    data_majority = data[data['income'] == 0]
    data_minority = data[data['income'] == 1]

    #   print(f"{len(data_majority)=}")
    #   print(f"{len(data_minority)=}")

    data_minority_upsampled = resample(data_minority, replace=True, n_samples=len(data_majority), random_state=5)

    data_balanced = pd.concat([data_majority, data_minority_upsampled])

    from sklearn.model_selection import train_test_split

    y = data_balanced['income']
    X = data_balanced.drop('income',axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)
    normalizers = normalizer_(X_train, numerical_features)

    for feature in numerical_features:
        X_test[feature] = normalizers[feature].transform(X_test[[feature]])

    information_gain_feature = dict()
    for feature in X_train.columns:
        information_gain_feature[feature] = calculate_information_gain(pd.concat([X_train, y_train], axis=1), feature, 'income')
    sorted_dict_asc = dict(sorted(information_gain_feature.items(), key=lambda item: item[1], reverse=True))

    list(sorted_dict_asc.keys())

    top_k_features = list(sorted_dict_asc.keys())[:k]

    X_train_top_k = X_train[top_k_features]
    X_test_top_k = X_test[top_k_features]

    return X_train_top_k, y_train, X_test_top_k, y_test