import argparse
import tensorflow as tf
import numpy as np

from models import End_to_End_PathNet
from utils import *
from data_processing import *

import os


def get_command_line_args():
    parser = argparse.ArgumentParser(description="End-to-End PathNet for CIFAR classification")
    parser.add_argument("-L", "--layer", type=int, default=3,
                        help="number of layers of End-to-End PathNet default 3")
    parser.add_argument("-M", "--module", type=int, default=16,
                        help="number of modules in each layers default 16")
    parser.add_argument("-N", "--num_path", type=int, default=4,
                        help="maximum number of pathway between two layers 4")
    parser.add_argument("-hidden", type=int, default=20,
                        help="number of hidden units in each modules default 20")
    parser.add_argument("-s", "--source", type=str, choices = ['c1', 'c2'], default = 'c1',
                        help = 'source task to transfer. c1:c1->c2, c2:c2->c1 default c1')
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-5,
                        help="learning rate for learning default 1e-5")
    parser.add_argument("-b", "--batch", type=int, default=64,
                        help="size of bath default 16")
    parser.add_argument("-it", "--iteration", type=float, default=5000,
                        help="the number of iteration to stop traing default 5000")
    parser.add_argument("-r", "--repeat", type=int, default=10,
                        help="the number of how many time you want to repeat this experiment default 10")
    args = parser.parse_args()
    return args

def main():
    args = get_command_line_args()

    (X_train_c1, y_train_c1, X_test_c1, y_test_c1), (X_train_c2, y_train_c2, X_test_c2, y_test_c2) = cifar_classification()
    if args.source=='c1':
        source='c1'
        target='c2'
        X_train_source, y_train_source, X_test_source, y_test_source = X_train_c1, y_train_c1, X_test_c1, y_test_c1
        X_train_target, y_train_target, X_test_target, y_test_target = X_train_c2, y_train_c2, X_test_c2, y_test_c2
    else:
        source='c2'
        target='c1'
        X_train_source, y_train_source, X_test_source, y_test_source = X_train_c2, y_train_c2, X_test_c2, y_test_c2
        X_train_target, y_train_target, X_test_target, y_test_target = X_train_c1, y_train_c1, X_test_c1, y_test_c1
    data_train_source = Dataset(X_train_source, y_train_source)
    data_train_target = Dataset(X_train_target, y_train_target)
    
    BATCH_SIZE = args.batch
    OUTPUT_DIM = 5
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, 3072], name='x')
        y = tf.placeholder(tf.float32, [None, OUTPUT_DIM], name='y')
        transfer = tf.placeholder(tf.int32)

        Path = End_to_End_PathNet(num_layer=args.layer, num_module=args.module, num_hidden=args.hidden, num_path=args.num_path, output_dim=OUTPUT_DIM)

        outputs = Path(x, transfer)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=outputs)
        total_loss = tf.reduce_mean(loss)

        LR = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate=LR)
        train_opt = optimizer.minimize(total_loss)

        init = tf.global_variables_initializer()

    sess = tf.Session(graph=graph)
    sess.run(init)
    print('Graph Ready')
    
    max_iter = args.iteration


    print('Training start!')
    for s in range(args.repeat):
        sess.run(init)

        transfer_=0
        for epoch in range(1,max_iter+1):
            data, label = data_train_source.next_batch(BATCH_SIZE)
            sess.run(train_opt, feed_dict={x:data, y:label, LR:args.learning_rate, transfer:transfer_})

        outputs_ = sess.run(outputs, feed_dict={x:X_test_source, y:y_test_source, LR:args.learning_rate, transfer:transfer_})
        correct = y_test_source.argmax(axis=-1)==outputs_.argmax(axis=-1)
        accuracy = sum(correct)/len(y_test_source)
        with open("results/CIFAR_"+source+"_to_"+target+"_source.txt", "a") as fp:
            fp.write(str(accuracy)+' ')
        print('Experiment {}'.format(s+1))
        print('source task('+source+'): {}'.format(accuracy))

        transfer_=1
        for epoch in range(1,max_iter+1):
            data, label = data_train_target.next_batch(BATCH_SIZE)
            sess.run(train_opt, feed_dict={x:data, y:label, LR:args.learning_rate, transfer:transfer_})

        outputs_ = sess.run(outputs, feed_dict={x:X_test_target, y:y_test_target, LR:args.learning_rate, transfer:transfer_})
        correct = y_test_target.argmax(axis=-1)==outputs_.argmax(axis=-1)
        accuracy = sum(correct)/len(y_test_target)
        with open("results/CIFAR_"+source+"_to_"+target+"_target.txt", "a") as fp:
            fp.write(str(accuracy)+' ')
        print('target task('+target+'): {}'.format(accuracy))
        print()
    
    
if __name__ == "__main__":
    main()
