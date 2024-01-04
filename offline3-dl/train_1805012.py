from NN import * 
from preprocess import * 
from model_helpers import * 

X_train,y_train,X_test,y_test = preprocess_EMNIST() 

lrs = [5e-3,1e-3,5e-4,1e-4]
n_epochs = 10

for learning_rate in lrs: 
    model1 = NN(Linear(28*28,1024),ReLU(),Dropout(0.3),Linear(1024,26),Softmax())
    model3 = NN(Linear(28*28,512),ReLU(),Dropout(.2),Linear(512,512),ReLU(),Dropout(0.1),Linear(512,26),Softmax())
    model4 = NN(Linear(28*28,1024),ReLU(),Dropout(0.2),Linear(1024,512),ReLU(),Dropout(0.2),Linear(512,26),Softmax())
    models = [model1,model3,model4]
    for model in models:
        print(f"{learning_rate=}\n")
        print(f"{model=}\n")
        train_loss,val_loss,train_eval,val_eval = model.train(loss=cross_entropy, 
                                                      loss_grad=cross_entropy_grad, 
                                                      X=X_train, y=y_train, 
                                                      epochs=n_epochs, 
                                                      batch_size=2048, 
                                                      learning_rate=learning_rate,
                                                      eval_function=eval,
                                                      best_model_comparator=best_model_comparator,  
                                                      load_best_model_at_end=True,
                                                      save_best_model_at_each_epoch=False) 
        plot(train_loss,val_loss,train_eval,val_eval)
        eval_ = eval(model,X_test,y_test,batch_size=4096,plot_cm=True)
        acc, macro_f1 = eval_["Accuracy"], eval_["Macro_F1"]
        print(f"Test Accuracy: {acc}  Test Macro_F1: {macro_f1}")
        model.save(f"{model.model_name()}+_{learning_rate=}.pkl")
        print("\n ==================x================== \n")
        
        