from NN import * 
from preprocess import * 
from model_helpers import * 
import sys 

if len(sys.argv) < 2 :
    print("Usage: test_1805012.py <model_name>") 
    exit(-1)

X_train,y_train,X_test,y_test = preprocess_EMNIST() 

model = NN() 
model.load(sys.argv[1])
print(eval(model,X_test,y_test,plot_cm=True))