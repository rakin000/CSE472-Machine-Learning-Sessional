import numpy as np
rand = np.random.default_rng(42)

class Layer:
    def __init__(self):
        pass 

    def __call__(self,x,train=False):
        pass 

    def backward(self,out_grad,learning_rate):
        pass 

class Linear(Layer):
    def __init__(self,fan_in,fan_out,seed=42):
        self.fan_in = fan_in 
        self.fan_out = fan_out

        #xavier init 
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.weights = np.random.uniform(-limit, limit, size=(fan_out, fan_in))
        self.bias = np.ones((fan_out,1))

    def __call__(self,x,train=False):
        self.input = x
        return np.dot(self.weights,self.input) + self.bias
    

    def __repr__(self):
        pass 

    def backward(self, out_grad, learning_rate ):
        wgrad = np.dot(out_grad, self.input.T) / np.size(out_grad, axis=1) # mean  
        bgrad = np.mean(out_grad, axis=1, keepdims=True) 
        inputgrad = np.dot(self.weights.T, out_grad)

        self.weights -= learning_rate * wgrad 
        self.bias -= learning_rate * bgrad 

        # print(f"{self.wgrad}")
        # print(f"{self.bgrad}")

        return inputgrad 

    def reset_grad(self):
        self.wgrad = np.zeros((self.fan_in,self.fan_out))
        self.bgrad = np.zeros((1,self.fan_out)) 

class Softmax(Layer):
    def __call__(self, input, train=False):
        self.input = input
        tmp = input - np.max(input, axis=0)  
        tmp = np.exp(tmp)
        self.output = tmp / np.sum(tmp, axis=0, keepdims=True)
        return self.output 
    def backward(self, out_grad, learning_rate):
        assert out_grad.shape == self.input.shape
        n = np.size(self.output, axis=0) 
        # grad = np.hstack([ np.dot( (np.identity(n) - input )*input.T, out_grad) for input in self.input.T  ])
        # grad = np.hstack([np.dot( (np.identity(n) - self.input[:,i:i+1].T)*self.input[:,i:i+1], out_grad[:,i:i+1] ) for i in range(np.size(self.input,axis=1)) ])
         # Modify the backward calculation for improved numerical stability
        grad = np.hstack([
            np.dot( (np.identity(n) - self.output[:, i:i+1].T) * self.output[:, i:i+1] , out_grad[:, i:i+1])
            for i in range(np.size(self.input, axis=1))
        ])
         #np.dot((np.identity(n)-self.output.T) * self.output, out_grad)  
        return grad
    
class Activation(Layer):
    def __init__(self,activation,activation_prime):
        self.activation = activation 
        self.activation_grad = activation_prime

    def __call__(self, input, train=False):
        self.input = input 
        return self.activation(self.input) 
    
    def backward(self, out_grad, learning_rate):
        return np.multiply(out_grad, self.activation_grad(self.input))
    
class Tanh(Activation):
    def __init__(self):
        super().__init__(self.tanh,self.tanh_grad) 
    def tanh(self,x):
        return np.tanh(x)
    def tanh_grad(self,x):
        return 1-np.tanh(x)**2 

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(self.sigmoid,self.sigmoid_grad) 
    
    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))
    def sigmoid_grad(self,x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x)) 
    
class ReLU(Activation):
    def __init__(self):
        super().__init__(self.relu,self.relu_grad)

    def relu(self,x):
        return np.maximum(0,x)
    def relu_grad(self,x):
        return np.where(x > 0, 1, np.where(x < 0, 0, 0.5))

class Dropout(Layer):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def __call__(self, x, train=False):
        if train:
            self.mask = (np.random.rand(*x.shape) < (1 - self.dropout_rate)) / (1 - self.dropout_rate)
            # print(self.mask)
            return x * self.mask
        else:
            return x

    def backward(self, grad, learning_rate):
        return grad * self.mask if self.mask is not None else grad
    

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_grad(y_true, y_pred): # wrt y_pred
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def cross_entropy(y_true,y_pred,epsilon=1e-15):
    y_pred = np.clip( y_pred, epsilon, 1.0-epsilon)
    return np.mean(-y_true*np.log(y_pred)) 

def cross_entropy_grad(y_true,y_pred,epsilon=1e-15):
    y_pred = np.clip( y_pred, epsilon, 1.0-epsilon)
    return (-y_true/y_pred) / len(y_true)  

import pickle

class NN:
    def __init__(self, *layers):
        self.layers : Layer = [] 
        for layer in layers:
            self.layers += [layer]
        
    def __call__(self,input,train=False):
        for layer in self.layers:
            input = layer(input,train=train)
        return input 
    def backward(self, ):
        pass 
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            loaded_instance = pickle.load(file)
        self.__dict__.update(loaded_instance.__dict__)
    
    def train_minibatch(self,):
        pass 
    
    def train(self, loss, loss_grad, X, y, epochs = 1000, batch_size = 8, learning_rate = 0.001, verbose=True):
        shuffled_indices = np.arange(len(X))
        np.random.shuffle(shuffled_indices) 

        X = X[shuffled_indices]
        y = y[shuffled_indices] 

        for epoch in range(epochs):
            error = 0 

            for i in range((len(X)+batch_size-1)//batch_size):
                X_batch = X[i:i+batch_size].T
                y_batch = y[i:i+batch_size].T

                y_pred = self.__call__(X_batch,train=True)
                error += loss(y_batch,y_pred)

                grad = loss_grad(y_batch,y_pred)

                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)
            error *= batch_size/len(X)
            if verbose:
                print(f"{epoch=}, {error=}")

    def eval(self, X,y,batch_size=1):
        corr = 0 
        for i in range((len(X)+batch_size-1)//batch_size):
            x_ = X[i:i+batch_size].T
            y_ = y[i:i+batch_size].T

            y_pred = self.__call__(x_)
            corr += np.sum(np.argmax(y_pred,axis=0) == np.argmax(y_,axis=0)) 
        # for x,y in zip(X,y):
        #     y_pred = self.__call__(x)
        #     corr += 1 if np.argmax(y_pred) == np.argmax(y) else 0 
        
        return corr/len(X) 
    
