# White Box Network : obtaining a right composition ordering of functions
Official github code implementation of the paper "White Box Network : obtaining a right composition ordering of functions" with python=3.7.0 and tensorflow=1.13.1

## Authors
Anonymous submission

## Requirements
This code is written in python 3, detailed requirements can be referred in requirements.txt. 
```
pip install -r requirements.txt
```

## Ladder Logic Diagram task
Run ladder_logic_diagram.py. Options has default settings as follows:  
-layer : default 4 : number of layers of WBN  
-d, --duplicate : default 3 : number of how many times do you want to duplicate given functions in each layer  
-r, --regression : default 1 : pinitial L1 regularization parameter  
-lld : default 1: ladder logic diagram data number, you can choose 1,2 or 3. 1 means the data is from LLD1  
-l, --learning_rate : default 1e-3 : learning rate for learning  
-b, --batch : default 64 : size of batch  
-i, --iteration : default 100000 : maximum iteration number  
-th, --threshold : default 0.005 : threshold of error for changing regression parameter  
-p, --patience : default 10 : early stopping patience  
  
For lld1 and lld2 recommended layer number is 4, duplicate number is 3. For lld3 recommended layer number is 5, duplicate number is 4, learning_rate is 5e-4

                        
## binary MNIST task (end-to-end PathNet, tranfer)
Run binary_mnist_classification.py. Options has default settings as follows:  
-layer : default 3 : number of layers of end-to-end PathNet  
-M, --module: default 10 : number of modules in each layers  
-N, --num_path : default=3 : maximum number of pathway between two layers  
-hidden : default 20 : number of hidden units in each modules  
-s, --source : default (1, 3) : tuple of two ditits to classify in source task  
-t, --target : default (7, 8) : tuple of two ditits to classify in source task default (7, 8)  
-l, --learning_rate : default 1e-5 : learning rate for learning  
-b, --batch : default 16 : size of batch  
-th, --threshold : default 0.998 : threshold training accuracy to stop traing  
-r, --repeat : default 10 : the number of how many time you want to repeat this experiment  
  
    
## CIFAR classification task (end-to-end PathNet, tranfer)
Run cifar_classification.py. Options has default settings as follows:  
-layer : default 3 : number of layers of end-to-end PathNet  
-M, --module: default 16 : number of modules in each layers  
-N, --num_path : default=4 : maximum number of pathway between two layers  
-hidden : default 20 : number of hidden units in each modules  
-s, --source : default 'c1' : choose one of 'c1' and 'c2'. This is source task to transfer. c1:c1->c2, c2:c2->c1  
-l, --learning_rate : default 1e-5 : learning rate for learning  
-b, --batch : default 64 : size of batch  
-i, --iteration : default 100000 : maximum iteration number  
-it : --iteration : default=5000 : the number of iteration to stop traing  
-r, --repeat : default 10 : the number of how many time you want to repeat this experiment  
