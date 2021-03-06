"""
Simulation of just one stock price.
"""

import numpy as np
from stock_base import BaseStock

class Stock1(BaseStock):
    
    def __init__(self):
        self.d = 0.01
        self.derivate = np.random.uniform(-self.d,self.d)
        self.price = np.random.uniform()*100
        self.probability_of_change = 0.1
        
    def step(self, action):
        #This model is just a "complex" stochastic model with hidden variables. 
        #It is not intended to be an effective model of a stock price, just a model which we can work with
        #print "derivate: ", self.derivate        
        
        self.derivate = self.derivate + np.random.uniform(-1,1) *self.derivate  * 0.1 

        
        if np.random.uniform() <= self.probability_of_change:        
            #print "unespected event!"            
            self.probability_of_change *= 1.1
            self.derivate = np.random.uniform(-self.d,self.d)

        
        if np.random.uniform() <= self.probability_of_change*0.1: 
            #print "reset probability of change!"
            self.probability_of_change = 0.1
        
        if abs(self.derivate) > self.d:
            
            self.probability_of_change * 1.3
            
        self.price += self.derivate + np.random.uniform(-1,1) * self.derivate * 2
        if self.price < 0:
            self.price = 0.01
        #print "price", self.price
        return action * self.price
    
    def getState(self):
        return [self.price]
        
    def getEnvironmentInformation(self):
        #dimension of state, dimension of action, dimension of reward
        return (1,1,1)
    
    def reset(self):
        self.derivate = np.random.uniform(-1,1)
        self.price = np.random.uniform(10)
        self.probability_of_change = 0.01
