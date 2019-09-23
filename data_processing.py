import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils import *


def ladder_logic_data_1(num_inputs=8, train_size=1000, test_size=500):
        X_train = np.random.randint(2, size=(train_size, num_inputs))
        y_train = Multiplication_logit(Addition_logit(Multiplication_logit(X_train[:,0], X_train[:,1]), X_train[:,2]), Addition_logit(X_train[:,3], X_train[:,5]))
        y_train = y_train.reshape(-1, 1)

        X_test = np.random.randint(2, size=(test_size, num_inputs))
        y_test = Multiplication_logit(Addition_logit(Multiplication_logit(X_test[:,0], X_test[:,1]), X_test[:,2]), Addition_logit(X_test[:,3], X_test[:,5]))
        y_test = y_test.reshape(-1,1)
        return X_train, y_train, X_test, y_test
 
    
def ladder_logic_data_2(num_inputs=8, train_size=1000, test_size=500):
    X_train = np.random.randint(2, size=(train_size, num_inputs))
    y1 =  Addition_logit(X_train[:,0], X_train[:,3])
    y2 = Addition_logit(Addition_logit(X_train[:,0], X_train[:,3]), Addition_logit(X_train[:,1], X_train[:,2]))
    y3 = Addition_logit(Multiplication_logit(1-X_train[:,3], y2), Multiplication_logit(X_train[:,0], 1-y2))
    y_train = np.concatenate([y1.reshape(-1,1), y2.reshape(-1,1), y3.reshape(-1,1)], axis=-1)
    
    X_test = np.random.randint(2, size=(test_size, num_inputs))
    y1 =  Addition_logit(X_test[:,0], X_test[:,3])
    y2 = Addition_logit(Addition_logit(X_test[:,0], X_test[:,3]), Addition_logit(X_test[:,1], X_test[:,2]))
    y3 = Addition_logit(Multiplication_logit(1-X_test[:,3], y2), Multiplication_logit(X_test[:,0], 1-y2))
    y_test = np.concatenate([y1.reshape(-1,1), y2.reshape(-1,1), y3.reshape(-1,1)], axis=-1)
    return X_train, y_train, X_test, y_test

def ladder_logic_data_3(num_inputs=8, train_size=1000, test_size=500):
    X_train = np.random.randint(2, size=(train_size, num_inputs))
    y1 =  Addition_logit(X_train[:,0], X_train[:,3])
    y2 = Multiplication_logit(X_train[:,1], X_train[:,2])
    y3 = Addition_logit(Multiplication_logit(X_train[:,4], y1), Multiplication_logit(1-X_train[:,4], y2))
    y4 = Multiplication_logit(Multiplication_logit(y1, 1-y2), Addition_logit(Multiplication_logit(1-X_train[:,5], y3), Multiplication_logit(X_train[:,5], 1-y3)))
    y_train = np.concatenate([y1.reshape(-1,1), y2.reshape(-1,1), y3.reshape(-1,1), y4.reshape(-1,1)], axis=-1)
    
    X_test = np.random.randint(2, size=(test_size, num_inputs))
    y1 =  Addition_logit(X_test[:,0], X_test[:,3])
    y2 = Multiplication_logit(X_test[:,1], X_test[:,2])
    y3 = Addition_logit(Multiplication_logit(X_test[:,4], y1), Multiplication_logit(1-X_test[:,4], y2))
    y4 = Multiplication_logit(Multiplication_logit(y1, 1-y2), Addition_logit(Multiplication_logit(1-X_test[:,5], y3), Multiplication_logit(X_test[:,5], 1-y3)))
    y_test = np.concatenate([y1.reshape(-1,1), y2.reshape(-1,1), y3.reshape(-1,1), y4.reshape(-1,1)], axis=-1)
    return X_train, y_train, X_test, y_test

def binary_mnist_classification(train_size=3200, classification_digits=(1, 3)):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mnist_x = mnist.train.images
    mnist_y = mnist.train.labels
    x_list=[]
    y_list=[]
    for i in range(55000):
        if mnist_y[i][classification_digits[0]]==1:
            x_list.append(mnist_x[i])
            y_list.append(0)
        if mnist_y[i][classification_digits[1]]==1:
            x_list.append(mnist_x[i])
            y_list.append(1)
    X_train, y_train = np.array(x_list), np.array(y_list)
    X_train, y_train = X_train[:train_size], y_train[:train_size].reshape(-1, 1)
    
    mnist_x = mnist.test.images
    mnist_y = mnist.test.labels
    x_list=[]
    y_list=[]
    for i in range(10000):
        if mnist_y[i][classification_digits[0]]==1:
            x_list.append(mnist_x[i])
            y_list.append(0)
        if mnist_y[i][classification_digits[1]]==1:
            x_list.append(mnist_x[i])
            y_list.append(1)
    X_test, y_test = np.array(x_list), np.array(y_list).reshape(-1, 1)
    return X_train, y_train, X_test, y_test

def cifar_classification():
    from tensorflow.keras.datasets.cifar10 import load_data
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train_cifar = x_train.reshape(-1, 32*32*3)/255.0
    x_test_cifar = x_test.reshape(-1, 32*32*3)/255.0
    y_train_cifar = np.zeros((len(y_train), 10))
    y_test_cifar = np.zeros((len(y_test), 10))
    for i in range(len(y_train)):
        y_train_cifar[i][y_train[i]] = 1.0
    for i in range(len(y_test)):
        y_test_cifar[i][y_test[i]] = 1.0
    X_train_c1 = np.array([x_train_cifar[i] for i in range(len(x_train_cifar)) if y_train[i]<5])
    X_train_c2 = np.array([x_train_cifar[i] for i in range(len(x_train_cifar)) if y_train[i]>=5])
    y_train_c1 = np.array([y_train_cifar[i] for i in range(len(x_train_cifar)) if y_train[i]<5])[:,:5]
    y_train_c2 = np.array([y_train_cifar[i] for i in range(len(x_train_cifar)) if y_train[i]>=5])[:,5:]

    X_test_c1 = np.array([x_test_cifar[i] for i in range(len(x_test_cifar)) if y_test[i]<5])
    X_test_c2 = np.array([x_test_cifar[i] for i in range(len(x_test_cifar)) if y_test[i]>=5])
    y_test_c1 = np.array([y_test_cifar[i] for i in range(len(x_test_cifar)) if y_test[i]<5])[:,:5]
    y_test_c2 = np.array([y_test_cifar[i] for i in range(len(x_test_cifar)) if y_test[i]>=5])[:,5:]
    return (X_train_c1, y_train_c1, X_test_c1, y_test_c1), (X_train_c2, y_train_c2, X_test_c2, y_test_c2)