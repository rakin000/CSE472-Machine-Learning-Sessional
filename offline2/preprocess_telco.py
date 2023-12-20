import pandas as pd
import opendatasets as od
from data_preprocessor_helpers import * 
from sklearn.model_selection import train_test_split


def preprocess(k=40):
    od.download('https://www.kaggle.com/datasets/blastchar/telco-customer-churn')
    data = pd.read_csv('telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data.drop(columns=['customerID'],inplace=True)

    multi_category_features = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies','Contract','PaymentMethod',]
    binary_category_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn'] #,
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    category_features = binary_category_features+multi_category_features

    data["TotalCharges"] = data["TotalCharges"].replace(" ", 0.0)

    label_imapping = label_encoder_(data, binary_category_features)


    data = onehot_encoder_(data, multi_category_features)

    from sklearn.utils import resample

    data_majority = data[data['Churn'] == 0]
    data_minority = data[data['Churn'] == 1]

    print(f"{len(data_majority)=}")
    print(f"{len(data_minority)=}")

    data_minority_upsampled = resample(data_minority, replace=True, n_samples=len(data_majority), random_state=5)

    data_balanced = pd.concat([data_majority, data_minority_upsampled])


    

    y = data_balanced['Churn']
    X = data_balanced.drop('Churn',axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)

    normalizers = normalizer_(X_train, numerical_features)
    for feature in numerical_features:
        X_test[feature] = normalizers[feature].transform(X_test[[feature]])

    information_gain_feature = dict()
    for feature in X_train.columns:
        information_gain_feature[feature] = calculate_information_gain(pd.concat([X_train, y_train], axis=1), feature, 'Churn')

    sorted_dict_asc = dict(sorted(information_gain_feature.items(), key=lambda item: item[1], reverse=True))

    top_k_features = list(sorted_dict_asc.keys())[:k]
    top_k_features

    X_train_top_k = X_train[top_k_features]
    X_test_top_k = X_test[top_k_features]
    return X_train_top_k, y_train, X_test_top_k, y_test 
