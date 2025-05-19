import numpy as np
import tensorflow as tf


class WhiteBoxNet(object):
    def __init__(self, output_dim, layer_size, unitary_func_list, binary_func_list):
        self.output_dim = output_dim
        self.layer_size = layer_size
        self.unitary_func_list = unitary_func_list
        self.binary_func_list = binary_func_list
    
    def __call__(self, inputs):
        
        hidden = inputs
        W=[]
        for layer in range(self.layer_size):
            hidden, W_= self.build_layer('layer_{}'.format(layer), hidden, self.unitary_func_list, self.binary_func_list)
            W.append(W_)
        outputs, W_ = self.output_layer(hidden)
        W.append(W_)
        return outputs, W
    
    def build_layer(self, name, inputs, unitary_func_list, binary_func_list):
        # Build EQN layer(linear+equation)
        with tf.variable_scope(name):
            
            num_unitary = len(unitary_func_list)
            num_binary = len(binary_func_list)
            num_hidden = num_unitary + 2*num_binary
            
            input_dim = inputs.shape[-1]
            W = tf.get_variable(name=name, shape=(input_dim, num_hidden))
            temperature = 1e-4
            W = tf.nn.softmax(W/temperature, axis=0)
            #gamma = 10 + tf.nn.softplus(tf.get_variable(name=name+'_gamma', shape=(num_hidden)))
            #W_g = W**gamma
            #W_ = W_g/tf.reduce_sum(W_g, axis=0)
            W_ = W/tf.reduce_sum(W, axis=0)
            
            x_relocated = tf.matmul(inputs, W_)

            outputs = []
            for i in range(num_unitary):
                outputs.append(unitary_func_list[i](tf.reshape(x_relocated[:, i], (-1, 1))))
            for i in range(num_binary):
                outputs.append(binary_func_list[i](tf.reshape(x_relocated[:, num_unitary+2*i], (-1, 1)), tf.reshape(x_relocated[:, num_unitary+2*i+1], (-1, 1))))
            outputs = tf.concat(outputs, axis=-1)
            return outputs, W_
        
    def output_layer(self, last_hidden):
        # Last layer for outputs
        with tf.variable_scope('outputs'):
            W = tf.get_variable(name='output_W', shape=(last_hidden.shape[-1], self.output_dim))
            temperature = 1e-4
            W = tf.nn.softmax(W/temperature, axis=0)
            #gamma = 10 + tf.nn.softplus(tf.get_variable(name='outputs_'+'_gamma', shape=(self.output_dim)))
            #W_g = W**gamma
            #W_ = W_g/tf.reduce_sum(W_g, axis=0)
            W_ = W/tf.reduce_sum(W, axis=0)
            outputs = tf.matmul(last_hidden, W_)
            return outputs, W_
        
        
class End_to_End_PathNet(object):
    def __init__(self, num_layer=3, num_module=10, num_hidden=20, num_path=3, output_dim=2):
        self.num_layer = num_layer
        self.num_module = num_module
        self.num_hidden = num_hidden
        self.num_path = num_path
        self.output_dim = output_dim
        
    def __call__(self, inputs, transfer):
        with tf.variable_scope('pathnet', reuse = tf.AUTO_REUSE):
            hidden = inputs
            for i in range (self.num_layer):
                hidden_list = []
                for j in range(self.num_module):
                    hidden_list.append(tf.layers.dense(name='network{}_{}'.format(i,j), inputs=hidden, units=self.num_hidden, activation=tf.nn.relu, trainable=True))
                hidden_matrix = tf.stack(hidden_list)

                W = tf.get_variable(name='path_weight_{}'.format(i), shape=(self.num_path, self.num_module))
                temperature = tf.math.exp(3+10*tf.nn.softplus(tf.get_variable(name='temperature_{}'.format(i), shape=(1))))
                W = tf.nn.softmax(W*temperature)
                #gamma = 10 + tf.nn.softplus(tf.get_variable(name=name+'_gamma', shape=(num_hidden)))
                #W_g = W**gamma
                #W_ = W_g/tf.reduce_sum(W_g, axis=0)
                hidden = tf.einsum('jnk,ij->nk', hidden_matrix, W)

            outputs = tf.layers.dense(name='output_layer', inputs=hidden, units=self.output_dim, activation=None)


            hidden_t = inputs
            for i in range (self.num_layer):
                hidden_list_t = []
                W = tf.get_variable(name='path_weight_{}'.format(i), shape=(self.num_path, self.num_module))
                transfer_network_list = []
                for path in range(self.num_path):
                    transfer_network_list.append(tf.argmax(W[path]))
                for j in range(self.num_module):
                    if j in transfer_network_list:
                        hidden_list_t.append(tf.layers.dense(name='network_{}_{}'.format(i,j), inputs=hidden_t, units=self.num_hidden, activation=tf.nn.relu, trainable=False))
                    else:
                        hidden_list_t.append(tf.layers.dense(name='network_transfer_{}_{}'.format(i,j), inputs=hidden_t, units=self.num_hidden, activation=tf.nn.relu))
                hidden_matrix_t = tf.stack(hidden_list_t)

                W_t = tf.get_variable(name='path_weight_transfer{}'.format(i), shape=(self.num_path, self.num_module))
                temperature_t = tf.math.exp(3+10*tf.nn.softplus(tf.get_variable(name='temperature_transfer_{}'.format(i), shape=(1))))
                W_t = tf.nn.softmax(W_t*temperature_t)
                #gamma = 10 + tf.nn.softplus(tf.get_variable(name=name+'_gamma', shape=(num_hidden)))
                #W_g = W**gamma
                #W_ = W_g/tf.reduce_sum(W_g, axis=0)
                hidden_t = tf.einsum('jnk,ij->nk', hidden_matrix_t, W_t)

            outputs_t = tf.layers.dense(name='output_layer_transfer', inputs=hidden_t, units=self.output_dim, activation=None)

            final_outputs= tf.cond(transfer > 0, lambda: outputs_t, lambda: outputs)
        
        return final_outputs
