# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 10:24:42 2016

@author: hessam
"""
import numpy as np
import sklearn
from sklearn.neural_network import BernoulliRBM


X_train = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
rbm = BernoulliRBM(n_components=2,random_state=0, verbose=True)
rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.fit(X_train)
print(rbm.components_)
