import argparse
import tensorflow as tf

import numpy as np
from models import WhiteBoxNet
from utils import *
from data_processing import *


def get_command_line_args():
    parser = argparse.ArgumentParser(description="WhiteBoxNet for ladder logic diagrams")
    parser.add_argument("-layer", type=int, default=4,
                        help="number of layers of WBN default 4")
    parser.add_argument("-d", "--duplicate", type=int, default=3,
                        help="number of how many times do we duplicate given functions default 3")
    parser.add_argument("-r", "--regression", type=float, default=1,
                        help="initial L1 regularization parameter default 1")
    parser.add_argument("-lld", "--ladder_logic_diagram", type=int, default=1,
                        help="ladder logic diagram data number default 1")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-3,
                        help="learning rate for learning default 1e-3")
    parser.add_argument("-b", "--batch", type=int, default=64,
                        help="size of bath default 64")
    parser.add_argument("-i", "--iteration", type=int, default=100000,
                        help="maximum iteration number default 100000")
    parser.add_argument("-p", "--patience", type=int, default=10,
                        help="early stopping patience default 10")
    args = parser.parse_args()
    return args

def main():
    args = get_command_line_args()
    
    INPUT_DIM = 8
    if args.ladder_logic_diagram == 1:
        OUTPUT_DIM = 1
    if args.ladder_logic_diagram == 2:
        OUTPUT_DIM = 3
    if args.ladder_logic_diagram == 3:
        OUTPUT_DIM = 4
        
    tf.logging.set_verbosity(tf.logging.ERROR)
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, INPUT_DIM], name='x')
        y = tf.placeholder(tf.float32, [None, OUTPUT_DIM], name='y')
        reg = tf.placeholder(tf.float32, [])
        
        unitary_func_list = []
        binary_func_list = []
        for i in range(args.duplicate):
            unitary_func_list.append(Identity)
            unitary_func_list.append(Not)
            binary_func_list.append(Addition_logit)
            binary_func_list.append(Multiplication_logit)
            
        WBN = WhiteBoxNet(output_dim = OUTPUT_DIM, layer_size=args.layer, unitary_func_list=unitary_func_list, binary_func_list=binary_func_list)
        
        outputs, W = WBN(x)

        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=reg, scope=None)
        weights = tf.trainable_variables()  
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights) 

        MSE = tf.losses.mean_squared_error(labels=y, predictions=outputs)
        total_MSE = tf.reduce_sum(MSE)
        total_loss = total_MSE + regularization_penalty

        LR = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate=LR)
        train_opt = optimizer.minimize(total_loss)
        init = tf.global_variables_initializer()
    
    sess = tf.Session(graph=graph)
    sess.run(init)
    print('Graph Ready')
    
    if args.ladder_logic_diagram == 1:
        X_train, y_train, X_test, y_test = ladder_logic_data_1()
    if args.ladder_logic_diagram == 2:
        X_train, y_train, X_test, y_test = ladder_logic_data_2()
    if args.ladder_logic_diagram == 3:
        X_train, y_train, X_test, y_test = ladder_logic_data_3()
    X_valid, y_valid = X_train[-int(len(X_train)*0.2):], y_train[-int(len(X_train)*0.2):]
    data_set = Dataset(X_train[:int(len(X_train)*0.8)], y_train[:int(len(X_train)*0.8)])
    
    max_iter = args.iteration
    learning_rate = args.learning_rate
    alpha_init = args.regression
    PRINT = 1000
    
    
    alpha=alpha_init
    count=0
    loss_min = 1000
    print('Training start!!')
    for epoch in range(1,max_iter+1):
        data, label = data_set.next_batch(args.batch)
        sess.run(train_opt, feed_dict={x:data, y:label, LR:learning_rate, reg:alpha})
        if epoch%PRINT==0:
            loss_ = sess.run(total_MSE, feed_dict={x:X_valid, y:y_valid, LR:learning_rate, reg:alpha})
            print('Validation loss : {}'.format(loss_))
            if loss_ < 0.005:
                alpha = 1e-9
                
        if alpha<alpha_init and loss_<loss_min:
            loss_min = loss_
            count = 0
        elif alpha<alpha_init and loss_>loss_min:
            count +=1
            
        if count==args.patience:
            break
    print('Test loss: {}'.format(sess.run(total_MSE, feed_dict={x:X_test, y:y_test, LR:learning_rate, reg:alpha})))
    print()
    W_ = sess.run(W, feed_dict={x:X_valid, y:y_valid, LR:learning_rate, reg:alpha})
    print('Expected Equation by WBN is')
    for i in range(W_[-1].shape[1]):
        print('y{} = '.format(i+1)+Expected_Equation_LLD(W_[:-1]+[W_[-1][:,i].reshape(-1,1)], index=0, layer=args.layer, unitary_num=2*args.duplicate, binary_num=2*args.duplicate))

if __name__ == "__main__":
    main()
