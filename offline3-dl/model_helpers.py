def plot(train_loss,val_loss,train_eval,val_eval):
    import matplotlib.pyplot as plt
    import pandas as pd 
    epochs = range(1, len(train_loss) + 1)

    # loss graph
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'y-', label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    train_eval = pd.DataFrame(train_eval)
    val_eval = pd.DataFrame(val_eval)
    # accuracy graph
    plt.plot(epochs, list(train_eval["Accuracy"]), 'b-', label='Training Accuracy')
    plt.plot(epochs, list(val_eval["Accuracy"]), 'y-', label='Validation Accuracy')

    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # macro f1 graph 
    plt.plot(epochs, list(train_eval["Macro_F1"]), 'b-', label='Training Macro F1')
    plt.plot(epochs, list(val_eval["Macro_F1"]), 'y-', label='Validation Macro F1')

    plt.title('Training and Validation Macro F1')
    plt.xlabel('Epochs')
    plt.ylabel('Macro F1')
    plt.legend()

    plt.show()

def eval(model,X,y,batch_size=1024,plot_cm=False):
    num_batches = (len(X)+batch_size-1)//batch_size

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    import numpy as np


    y_true = y.T.argmax(axis=0)
    y_pred = np.concatenate([model(X[i*batch_size:(i+1)*batch_size].T).argmax(axis=0) for i in range(num_batches)])

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)

    # print(f'Macro F1 Score: {macro_f1}')
    # print(f'Accuracy: {accuracy}')
    if plot_cm:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 27), yticklabels=range(1, 27))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    return {"Accuracy":accuracy,"Macro_F1":macro_f1} 

def best_model_comparator(model1,model1_eval,model2,model2_eval):
    if model2 is None or model2_eval is None:
        return model1,model1_eval  
    if model1 is None or model1_eval is None:
        return model2,model2_eval
    
    if model1_eval["Macro_F1"] > model2_eval["Macro_F1"]:
        return model1,model1_eval 
    return model2,model2_eval 