import numpy as np 
import pandas as pd 

class LogisticRegression:
    def __init__(self, error_thres=0.5, learning_rate=0.01, n_iters=1000):
        self.error_thres = error_thres
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            err = np.mean((y_predicted-y) ** 2) # MSE
            if err < self.error_thres :
              break

            dw = np.dot(X.T, (y_predicted - y))
            # dw = dw / np.linalg.norm(dw)
            # db = (1 / n_samples) * np.sum(y_predicted - y)
            db = np.sum(y_predicted-y)
            # db = db / np.linalg.norm(db)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db



        # print(f"{X.T=}")
        # print(f"{np.dot(X.T,(y_predicted-y))}")

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

def accuracy(y_true, y_pred):
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy


def score(y_pred,y):  
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    cm = confusion_matrix(y, y_pred)
    # True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    accuracy = (TP+TN)/(TP+TN+FP+FN)
    true_positive_rate = TP / (TP + FN)
    true_negative_rate = TN / (TN + FP)
    positive_predictive_value = TP / (TP + FP)
    false_discovery_rate = FP / (TP + FP)
    f1 = f1_score(y, y_pred)


    print("LR classification accuracy:", accuracy)
    print(f'True Positive Rate: {true_positive_rate:.2f}')
    print(f'True Negative Rate: {true_negative_rate:.2f}')
    print(f'Positive Predictive Value: {positive_predictive_value:.2f}')
    print(f'False Discovery Rate: {false_discovery_rate:.2f}')
    print(f'F1 Score: {f1:.2f}')

if __name__ == "__main__":
  
    from preprocess_adult import preprocess

    X_train_top_k, y_train, X_test_top_k, y_test = preprocess() 

    regressor = LogisticRegression(error_thres=0)
    regressor.fit(X_train_top_k, y_train)
    y_pred = regressor.predict(X_test_top_k)

    print("Test dataset")
    score(y_pred,y_test) 

    y_pred = regressor.predict(X_train_top_k)
    
    print("Train dataset")
    score(y_pred,y_train)


    
