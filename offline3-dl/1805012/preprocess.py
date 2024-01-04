import numpy as np
def preprocess_EMNIST():
    from torchvision import datasets, transforms 

    train_validation_dataset = datasets.EMNIST(root='./data', 
                                           split='letters',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True,
                                        )


    independent_test_dataset = datasets.EMNIST(
                             root='./data',
                             split='letters',
                             train=False,
                             transform=transforms.ToTensor(),
                             )
    n_classes = len(train_validation_dataset.classes)-1
    n_datapoints = len(train_validation_dataset.targets)

    X_train = np.array(train_validation_dataset.data.reshape(-1,28*28))
    y_train = np.array([[1 if temp == i+1 else 0 for i in range(n_classes)] for temp in train_validation_dataset.targets])

    X_test = np.array(independent_test_dataset.data.reshape(-1,28*28))
    y_test = np.array([[1 if temp == i+1 else 0 for i in range(n_classes)] for temp in independent_test_dataset.targets])

    X_mean = np.mean(X_train,axis=0,keepdims=True)
    eps = 1e-8
    X_std = np.std(X_train,axis=0,keepdims=True) + eps 

    X_train = (X_train-X_mean)/X_std 
    X_test = (X_test-X_mean)/X_std 
    
    return X_train,y_train,X_test,y_test