"""
Pipeline for training and evaluating a neural network on the MNIST dataset

@author: Haoyang (Ryan) Li
"""

import time
import nn_utils


def main():
    # parse arguments/hyperparameters
    params = nn_utils.parse_params()
    
    # load the MNIST dataset
    mnist_dir = './MNISTdata.hdf5'
    mnist = nn_utils.load_mnist(mnist_dir)
    
    # initialization
    (model, model_grads) = nn_utils.init_model(mnist,params)
    
    # training the model
    print("\nStart training")
    print("---------------")
    time_start = time.time()
    model = nn_utils.nn_train(model, model_grads, params, mnist)
    time_end = time.time()
    print("Training Time : %8.4f (s)" % ( time_end-time_start ) )
    
    # testing the model
    print("\nStart testing")
    print("--------------")
    nn_utils.nn_test(model, params, mnist)

 
if __name__ == "__main__":
    main()