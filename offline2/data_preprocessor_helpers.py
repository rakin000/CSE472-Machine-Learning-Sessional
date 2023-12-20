from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def label_encoder_(df, label_features):
  label_imapper = dict()
  le = LabelEncoder()
  for feature in label_features:
    df[feature] = le.fit_transform(df[feature])
    label_imapper[feature] = dict(zip(le.transform(le.classes_),le.classes_))
  return label_imapper


def onehot_encoder_(df, class_features):
  oe = OneHotEncoder()
  for feature in class_features:
    encoded = oe.fit_transform(df[[feature]])
    encoded_df = pd.DataFrame(encoded.toarray(), columns=oe.get_feature_names_out([feature]))
    df = pd.concat([df, encoded_df], axis=1).drop(feature, axis=1)
  # print(df.head())
  return df

def normalizer_(df, numerical_features):
  normalizers = dict()

  for feature in numerical_features:
    scaler = StandardScaler()
    df[feature] = scaler.fit_transform(df[[feature]])
    normalizers[feature] = scaler
  return normalizers

import pandas as pd
import numpy as np

def calculate_entropy(labels):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    probabilities = label_counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_information_gain(data, feature_name, target_name):
    total_entropy = calculate_entropy(data[target_name])

    unique_values = data[feature_name].unique()
    weighted_entropy = 0

    for value in unique_values:
        subset = data[data[feature_name] == value]
        subset_weight = len(subset) / len(data)
        subset_entropy = calculate_entropy(subset[target_name])
        weighted_entropy += subset_weight * subset_entropy

    information_gain = total_entropy - weighted_entropy
    return information_gain



