
from __future__ import print_function
import numpy as np
import sys
import time
from Util import *
from CLT_class import CLT

class MIXTURE_CLT():
    
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks
        

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def learn(self, dataset, n_components=2, max_iter=50, epsilon=1e-5):
        # For each component and each data point, we have a weight
        weights=np.zeros((n_components,dataset.shape[0]))

        # Randomly initialize the chow-liu trees and the mixture probabilities
        # Your code for random initialization goes here

        for itr in range(max_iter):
            #E-step: Complete the dataset to yield a weighted dataset
            # We store the weights in an array weights[ncomponents,number of points]
            
            #Your code for E-step here
            
            # M-step: Update the Chow-Liu Trees and the mixture probabilities
            
            #Your code for M-Step here

        
    
    """
        Compute the log-likelihood score of the dataset
    """
    def computeLL(self, dataset):
        ll = 0.0
        # Write your code below to compute likelihood of data
        #   Hint:   Likelihood of a data point "x" is sum_{c} P(c) T(x|c)
        #           where P(c) is mixture_prob of cth component and T(x|c) is the probability w.r.t. chow-liu tree at c
        #           To compute T(x|c) you can use the function given in class CLT
        return ll
    

    
'''
    After you implement the functions learn and computeLL, you can learn a mixture of trees using
    To learn Chow-Liu trees, you can use
    mix_clt=MIXTURE_CLT()
    ncomponents=10 #number of components
    max_iter=50 #max number of iterations for EM
    epsilon=1e-1 #converge if the difference in the log-likelihods between two iterations is smaller 1e-1
    dataset=Util.load_dataset(path-of-the-file)
    mix_clt.learn(dataset,ncomponents,max_iter,epsilon)
    
    To compute average log likelihood of a dataset w.r.t. the mixture, you can use
    mix_clt.computeLL(dataset)/dataset.shape[0]
'''

    
    


    