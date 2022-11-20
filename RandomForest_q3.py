from __future__ import print_function

from operator import itemgetter
from statistics import stdev
import numpy as np
import sys
import time
from Util import *
from CLT_class_random_forest import CLT
import math
import os

class RandomForest():
    
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks
        self.T_k=None
        self.p_k=None

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    

    def learn(self, dataset, n_components=2, R=1000):

        # For each component and each data point, we have a weight
        self.n_components = n_components
        # For RandomForest, weigts can be uniform - keeping them 1
        weights=np.ones((n_components, dataset.shape[0]))
        self.mixture_probs = [1/n_components] * n_components
        #print(self.mixture_probs)
        self.clt_list = [CLT for i in range(n_components)]
        bootstrapSet = []
        for k in range(n_components):
            # Bootstrap samples before lerning a tree
            bootstrapSet_k = []
            for i in range(dataset.shape[1]):
                randomSample = np.random.randint(0, dataset.shape[0])
                bootstrapSample = dataset[randomSample]
                bootstrapSet_k.append(bootstrapSample)

            bootstrapSet.append(bootstrapSet_k)


        bootstrapSet = np.array(bootstrapSet)

        for k in range(n_components):
            self.clt_list[k].learn(self, dataset=bootstrapSet[k], R=R)

        print(RandomForest.computeLL(self, dataset))
       

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
                t_prob.append(self.clt_list[k].getProb(self, dataset[i]))
            
            ll+=np.sum(np.multiply(self.mixture_probs[k], t_prob))/dataset.shape[0]
        return np.log(ll)


import csv

with open('logs_RF.csv', 'a') as f:
    writer = csv.writer(f)

    for i in os.listdir("dataset-1/"):

        print(i)

        d = {}
        print("Please wait, while we learn mixture models for dataset...", i)
        validateset = Util.load_dataset("dataset-1/"+i+"/"+i+".valid.data")

        # Latent variable can take values from [2, 5, 10, 20]
        for k_c in [2, 5, 10, 20]:
            for R in [10, 100, 200, 1000]:
                forest = RandomForest()
                forest.learn(validateset, n_components=k_c, R=R)
                print("Running on the validation set ",i, "when K=", k_c, " and R=", R)
                print("Valid set Log likelihood for", i, " = ", forest.computeLL(validateset), "when K=", k_c, "and R=", R)
                K_R=str(k_c)+" "+str(R)
                d[K_R] = forest.computeLL(validateset)

        s_d = dict(sorted(d.items(), key=itemgetter(1), reverse=True))

        traindataset = Util.load_dataset("dataset-1/" + i + "/" + i + ".ts.data")

        final_ll = []
        d_k = list(s_d.keys())[0]
        writer.writerow(['Valid', i, d_k.split(" ")[0], d_k.split(" ")[1], s_d[d_k]])
        print("Running on the Training set when the K = ", d_k.split(" ")[0], " and R=", d_k.split(" ")[1])

        for j in range(5):
            forest = RandomForest()
            forest.learn(traindataset, n_components=int(d_k.split(" ")[0]), R=int(d_k.split(" ")[1]))
            final_ll.append(forest.computeLL(traindataset))

        print("For train set", i, "Average =", np.mean(final_ll), "Standard Deviation =", np.std(final_ll))
        writer.writerow(['Train', i, np.mean(final_ll), "Standard Deviation", np.std(final_ll)])
        testdataset = Util.load_dataset("dataset-1/" + i + "/" + i + ".test.data")
        print("Test set log likelihood for ", i, " = ", forest.computeLL(testdataset))
        writer.writerow(['Test', i, forest.computeLL(testdataset)])


