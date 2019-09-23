import argparse
import tensorflow as tf
import numpy as np

from models import End_to_End_PathNet
from utils import *
from data_processing import *

import os

def get_command_line_args():
    parser = argparse.ArgumentParser(description="End-to-End PathNet for binary MNIST classification")
    parser.add_argument("-L", "--layer", type=int, default=3,
                        help="number of layers of End-to-End PathNet default 3")
    parser.add_argument("-M", "--module", type=int, default=10,
                        help="number of modules in each layers default 10")
    parser.add_argument("-N", "--num_path", type=int, default=3,
                        help="maximum number of pathway between two layers 3")
    parser.add_argument("-hidden", type=int, default=20,
                        help="number of hidden units in each modules default 20")
    parser.add_argument("-s", "--source", nargs=2, type=int, default=(1, 3),
                        help="tuple of two ditits to classify in source task default (1, 3)")
    parser.add_argument("-t","--target", nargs=2, type=int, default=(7, 8),
                        help="tuple of two ditits to classify in source task default (1, 3)")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-5,
                        help="learning rate for learning default 1e-5")
    parser.add_argument("-b", "--batch", type=int, default=16,
                        help="size of bath default 16")
    parser.add_argument("-th", "--threshold", type=float, default=0.998,
                        help="threshold training accuracy to stop traing default 0.998")
    parser.add_argument("-r", "--repeat", type=int, default=10,
                        help="the number of how many time you want to repeat this experiment default 10")
    args = parser.parse_args()
    return args

def main():
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    args = get_command_line_args()
    
    X_train_source, y_train_source, X_test_source, y_test_source = binary_mnist_classification(train_size=3200, classification_digits=args.source)
    X_train_target, y_train_target, X_test_target, y_test_target = binary_mnist_classification(train_size=3200, classification_digits=args.target)
    
    data_train_source = Dataset(X_train_source, y_train_source)
    data_train_target = Dataset(X_train_target, y_train_target)
    
    BATCH_SIZE = args.batch
    OUTPUT_DIM = 1
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, 784], name='x')
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
        
    sess = tf.Session(graph=graph, config=config)
    sess.run(init)
    print('Graph Ready')

    max_iter = 200000
    PRINT = 200
    learning_rate = args.learning_rate


    print('Training start!')
    for s in range(args.repeat):
        sess.run(init)

        transfer_=0
        generation = 0
        for epoch in range(1, 200000+1):
            data, label = data_train_source.next_batch(BATCH_SIZE)
            sess.run(train_opt, feed_dict={x:data, y:label, LR:learning_rate, transfer:transfer_})

            if epoch%PRINT==0:
                loss_, output_ = sess.run((total_loss, outputs), feed_dict={x:X_train_source, y:y_train_source, LR:learning_rate, transfer:transfer_})
                pred = [1 for i in range(3200) if (output_[i][0]>0 and y_train_source[i]==1) or (output_[i][0]<0 and y_train_source[i]==0)]
                accuracy = sum(pred)/3200
                generation += 1
                if accuracy>args.threshold:
                    with open("results/MNIST_{}vs{}_to_{}vs{}_source.txt".format(args.source[0], args.source[1], args.target[0], args.target[1]), "a") as fp:
                        fp.write(str(generation)+' ')
                    print('Experiment {}'.format(s+1))
                    print('source task({}vs{}): {} generations'.format(args.source[0], args.source[1], generation))
                    break

        transfer_=1
        generation = 0
        for epoch in range(1, 200000+1):
            data, label = data_train_target.next_batch(BATCH_SIZE)
            sess.run(train_opt, feed_dict={x:data, y:label, LR:learning_rate, transfer:transfer_})

            if epoch%PRINT==0:
                loss_, output_ = sess.run((total_loss, outputs), feed_dict={x:X_train_target, y:y_train_target, LR:learning_rate, transfer:transfer_})
                pred = [1 for i in range(3200) if (output_[i][0]>0 and y_train_target[i]==1) or (output_[i][0]<0 and y_train_target[i]==0)]
                accuracy = sum(pred)/3200
                generation += 1
                if accuracy>args.threshold:
                    with open("results/MNIST_{}vs{}_to_{}vs{}_target.txt".format(args.source[0], args.source[1], args.target[0], args.target[1]), "a") as fp:
                        fp.write(str(generation)+' ')
                    print('target task({}vs{}): {} generations'.format(args.target[0], args.target[1], generation))
                    break
        print()
    
    
if __name__ == "__main__":
    main()