import numpy as np


class Dataset:
    def __init__(self, inputs, outputs):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data_x = inputs
        self._data_y = outputs
        self._num_examples = len(inputs)
        pass


    @property
    def data_x(self):
        return self._data_x
    
    @property
    def data_y(self):
        return self._data_y

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data_x = np.array([self.data_x[i] for i in idx])   # get list of `num` random samples
            self._data_y = np.array([self.data_y[i] for i in idx])

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_x_rest_part = self.data_x[start:self._num_examples]
            data_y_rest_part = self.data_y[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data_x = np.array([self.data_x[i] for i in idx0]) # get list of `num` random samples
            self._data_y = np.array([self.data_y[i] for i in idx0])

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_x_new_part =  self._data_x[start:end]
            data_y_new_part =  self._data_y[start:end]
        
            batch_data_x = np.concatenate((data_x_rest_part, data_x_new_part), axis=0)
            batch_data_y = np.concatenate((data_y_rest_part, data_y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            batch_data_x = self._data_x[start:end]
            batch_data_y = self._data_y[start:end]
            
        return batch_data_x, batch_data_y
    
    
    
def Identity(x):
    return x

def Not(x):
    return 1-x

def Addition_logit(x1, x2):
    return x1+x2-x1*x2

def Multiplication_logit(x1, x2):
    return x1*x2



def Expected_Equation_LLD(W, index, layer, unitary_num, binary_num):
    index_new = np.argmax(W[layer], axis=0)[index]
    if layer==0:
        return  'X{}'.format(index_new+1)
    else:
        if index_new<unitary_num:
            if index_new%2==0:
                return Expected_Equation_LLD(W, index_new, layer-1, unitary_num, binary_num)
            elif index_new%2==1:
                return '(~'+Expected_Equation_LLD(W, index_new, layer-1, unitary_num, binary_num)+')'
        else:
            index_new1 = 2*index_new-unitary_num
            index_new2 = 2*index_new-unitary_num+1
            if (index_new-unitary_num)%2==0:
                return '('+Expected_Equation_LLD(W, index_new1, layer-1, unitary_num, binary_num)+'∨'+Expected_Equation_LLD(W, index_new2, layer-1, unitary_num, binary_num)+')'
            elif (index_new-unitary_num)%2==1:
                return '('+Expected_Equation_LLD(W, index_new1, layer-1, unitary_num, binary_num)+'∧'+Expected_Equation_LLD(W, index_new2, layer-1, unitary_num, binary_num)+')'
            
def Expected_Equation(W, index, layer, unitary_num, binary_num):
    index_new = np.argmax(W[layer], axis=0)[index]
    if layer==0:
        return  'X{}'.format(index_new+1)
    else:
        if index_new<unitary_num:
            return 'U{}('.format(index_new+1)+Expected_Equation(W, index_new, layer-1, unitary_num, binary_num)+')'
        else:
            index_new1 = 2*index_new-unitary_num
            index_new2 = 2*index_new-unitary_num+1
            return 'B{}('.format(index_new-unitary_num+1)+Expected_Equation(W, index_new1, layer-1, unitary_num, binary_num)+','+Expected_Equation(W, index_new2, layer-1, unitary_num, binary_num)+')'
