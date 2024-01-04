# %%
from NN import * 
from preprocess import *

# %%
X_train,y_train,X_test,y_test=preprocess_EMNIST() 

# %%
model = NN(Linear(28*28,1024),ReLU(),Dropout(0.3),Linear(1024,26),Softmax())

# %%
model

# %%
model.train(cross_entropy, cross_entropy_grad, X_train, y_train, epochs=50, batch_size=1024, learning_rate=5e-3, save_best_model_at_each_epoch=True) # learning_rate_scheduler=lambda epoch,i_lr:i_lr)

# %%
model.eval(X_test,y_test,batch_size=2048)

# %%
model2 = NN(Linear(28*28,1024,initializer="xavier"),Sigmoid(),Dropout(.3),Linear(1024,26),Softmax())
model2

# %%
model2.train(cross_entropy, cross_entropy_grad, X_train, y_train, epochs=50, batch_size=1024, learning_rate=5e-3, save_best_model_at_each_epoch=True) # learning_rate_scheduler=lambda epoch,i_lr:i_lr)

# %% [markdown]
# 

# %%
model2.eval(X_test,y_test,batch_size=4096)

# %%
model3 = NN(Linear(28*28,512),ReLU(),Dropout(.2),Linear(512,512),ReLU(),Dropout(0.1),Linear(512,26),Softmax())
model3

# %%
model3.train(cross_entropy, cross_entropy_grad, X_train, y_train, epochs=50, batch_size=1024, learning_rate=5e-3, save_best_model_at_each_epoch=True)

# %%
model3.eval(X_test,y_test,batch_size=4096)

# %%
model4 = NN(Linear(28*28,256),ReLU(),Dropout(0.02),Linear(256,512),ReLU(),Dropout(0.02),)


