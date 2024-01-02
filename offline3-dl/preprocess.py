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
    n_classes = len(train_validation_dataset.classes)
    n_datapoints = len(train_validation_dataset.targets)

    X_train = np.array(train_validation_dataset.data.reshape(-1,28*28))

    y_train = np.array([[1 if temp == i else 0 for i in range(n_classes)] for temp in train_validation_dataset.targets])

    X_test = np.array(independent_test_dataset.data.reshape(-1,28*28))
    y_test = np.array([[1 if temp == i else 0 for i in range(n_classes)] for temp in independent_test_dataset.targets])

    return X_train,y_train,X_test,y_test