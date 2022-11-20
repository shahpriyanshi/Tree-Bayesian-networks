from __future__ import print_function
import numpy as np
import sys
import time
from Util import *
from CLT_class import CLT
import math
import os
from operator import itemgetter

class MIXTURE_CLT():
    
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks
        self.T_k=None
        self.p_k=None

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    

    def learn(self, dataset, n_components=2, max_iter=5, epsilon=1e-5, train=0):

        self.n_components = n_components
        # For each component and each data point, we have a weight
        weights=np.zeros((dataset.shape[0], n_components))
        self.p_k=np.zeros((n_components, dataset.shape[0]))
        # Randomly initialize the chow-liu trees and the mixture probabilities
        # Your code for random initialization goes here
        randomWeights = np.random.rand(n_components)

        #lambda sum to 1
        self.mixture_probs = randomWeights/np.sum(randomWeights)
        #print(self.mixture_probs)
        
        # For storing the logliklihood from the previous iteration
        secondLL = 0

        # Initializing one CLT for each value of the latent variable
        self.clt_list = [CLT() for i in range(n_components)]

         
        # Learning each Chow-Liu Tree
        for k in range(n_components):
          self.clt_list[k].learn(dataset)

        self.T_k = []
        for i in range(dataset.shape[0]):
          ra = np.random.rand(n_components)
          ra /= np.sum(ra)
          self.T_k.append(ra)

        self.T_k = np.array(self.T_k)
        self.mixture_probs = np.array(self.mixture_probs)
        final_ll=0
        for itr in range(max_iter):
          for i in range(dataset.shape[0]):
              weights[i] = self.T_k[i] * self.mixture_probs
              weights[i]/=np.sum(weights[i])

          # print(weights)
          
          tou=np.array(np.sum(weights, axis=0))
          
          # print("t", self.T_k.shape)

          self.p_k = weights/tou
          
          # print("tou", self.p_k.shape)

          for k in range(n_components):

            # M-step: Update the Chow-Liu Trees and the mixture probabilities
            # Your code for M-Step here
            
            self.mixture_probs[k] = tou[k]/np.sum(weights)
            #print("mp", self.mixture_probs)
            self.clt_list[k].update(dataset, self.p_k.T[k])

          self.T_k=[]  
          for i in range(dataset.shape[0]):
            n_t_prob = [t.getProb(dataset[i]) for t in self.clt_list]   
            self.T_k.append(n_t_prob)

          self.T_k=np.array(self.T_k)
          # print("nn", self.T_k.shape)

          # Compare two consecutive log liklihoods. And if the difference is less than Epsilon, break/converge.
          # Since logliklihood is only going to increase, we can take difference accordingly

          
          if(itr == 0):
              secondLL = mix.computeLL(dataset)
          else:
              print("Entered in the iteration ... ", itr)
              firstLL = mix.computeLL(dataset)
              #print("loglikelihood", firstLL)
              if(abs(firstLL - secondLL) < epsilon):
                  print("Exiting because the increase in log likelihood value is less than epsilon ... ")
                  break
              secondLL = firstLL

    """
        Compute the log-likelihood score of the dataset
    """


    def computeLL(self, dataset):
        # Write your code below to compute likelihood of data
        #   Hint:   Likelihood of a data point "x" is sum_{c} P(c) T(x|c)
        #           where P(c) is mixture_prob of cth component and T(x|c) is the probability w.r.t. chow-liu tree at c
        #           To compute T(x|c) you can use the function given in class CLT

        ll=0.0

        # getProb gives one value for each example
        # getProb gives one value for each example

        for k in range(self.n_components):
            t_prob=[]
            for i in range(dataset.shape[0]):
                t_prob.append(self.clt_list[k].getProb(dataset[i]))
            
            ll+=np.sum(np.multiply(self.mixture_probs[k], t_prob))/dataset.shape[0]
  
        return np.log(ll)


import csv

with open('logs_em.csv', 'w') as f:
    writer = csv.writer(f)

    for i in os.listdir("dataset/"):

        print(i)

        d = {}
        print("Please wait, while we learn mixture models for dataset...", i)
        dataset = Util.load_dataset("dataset/"+i+"/"+i+".valid.data")

        # Latent variable can take values from [2, 5, 10, 20]
        for h_k in [2, 5, 10, 20]:
            print("Running on the validation set when the hidden variable can take up to", h_k,"values")
            mix = MIXTURE_CLT()
            mix.learn(dataset, n_components=h_k, max_iter=50, epsilon=1e-1, train=0)
            print("Valid set Log likelihood for", i, " = ", mix.computeLL(dataset), "when K=", h_k)
            d[h_k]=mix.computeLL(dataset)

        s_d = dict(sorted(d.items(), key=itemgetter(1), reverse=True))

        traindataset = Util.load_dataset("dataset/"+i+"/"+i+".ts.data")

        final_ll=[]
        d_k=list(s_d.keys())[0]
        writer.writerow(['Valid', i, d_k, s_d[d_k]])
        print("Running on the Training set when the K = ", d_k, "values")

        for j in range(5):
            mix = MIXTURE_CLT()
            mix.learn(traindataset, n_components=d_k, max_iter=1, epsilon=1e-1)
            final_ll.append(mix.computeLL(traindataset))
        print("For train set", i, "Average =", np.mean(final_ll), "Standard Deviation =", np.std(final_ll))
        writer.writerow(['Train', i, np.mean(final_ll), "Standard Deviation", np.std(final_ll)])
        testdataset = Util.load_dataset("dataset/"+i+"/"+i+".test.data")
        print("Test set log likelihood for ", i, " = ", mix.computeLL(testdataset))
        writer.writerow(['Test', i, mix.computeLL(testdataset)])

