import copy
class AdaBoostClassifier:
    def __init__(self, base_estimator, k=50):
        self.k = k
        self.base_estimator = base_estimator
        self.w = []
        self.z = []
        self.models = []
        self.epsilon = 1e-10


    def resample(self,X,y):
      n = len(X)
      # for _ in range(n):
      print(self.w)
      indices = np.random.choice(len(X), size=len(X), replace=True, p=self.w)

      return X.iloc[indices], y.iloc[indices]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.ones(n_samples) / n_samples  # Initialize weights uniformly

        for _ in range(self.k):
            indices = np.random.choice(len(X),size=len(X), replace=True, p=self.w)
            X_ = X.iloc[indices]
            y_ = y.iloc[indices]

            model = copy.deepcopy(self.base_estimator)
            model.fit(X_, y_)

            y_pred = model.predict(X)

            error = np.sum(self.w * (y_pred != y))

            # print(error)
            # if error > 0.5 :
            #   continue
            # print("here")

            error = max(error,self.epsilon)

            matching_predictions = (y_pred == y)  # Find where predictions match y
            self.w[matching_predictions] *= error / (1 - error)
            self.w /= np.sum(self.w)


            z_ = 0.5 * np.log((1.0 - error) / error)

            # w = w * np.exp(-z_ * y_ * y_pred)
            # w /= np.sum(w)
            self.z.append(z_)
            self.models.append(model)
        print(f"{self.models}")

    def predict(self, X):
        final_predictions = np.zeros(len(X))
        # print(len(self.models))
        for z_, model in zip(self.z, self.models):
            predictions = z_ * np.array(model.predict(X))
            final_predictions += predictions
        # print(f"{final_predictions=}")
        # print(f"{final_predictions.shape}")
        y_predicted_cls = [1 if i > 0.5 else 0 for i in final_predictions]
        return np.array(y_predicted_cls)


if __name__ == '__main__':
    from LogisticRegression import *
    from sklearn.metrics import accuracy_score
    from preprocess_credit import preprocess

    X_train_top_k, y_train, X_test_top_k, y_test = preprocess() 


    for k in [5,10,15,20]:
        regressor = LogisticRegression(error_thres=0.2)
        adaboost_classifier = AdaBoostClassifier(base_estimator=regressor,k=k)
        adaboost_classifier.fit(X_train_top_k, y_train)

        y_pred = adaboost_classifier.predict(X_test_top_k)
        accuracy = accuracy_score(y_test,y_pred)
        print(f"{k=}, Test Set Accuracy: {accuracy:.4f}")

        y_pred = adaboost_classifier.predict(X_train_top_k)
        accuracy = accuracy_score(y_train,y_pred)
        print(f"{k=}, Train Set Accuracy: {accuracy:.4f}")



