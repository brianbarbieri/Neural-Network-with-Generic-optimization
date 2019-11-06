# 1 hidden layer neural network with generic optimization algorithm
import numpy as np
import pandas as pd
import random
from itertools import permutations 
import matplotlib.pyplot as plt

class Individual:
    def __init__(self, w1=None, w2=None, b1=None, b2=None):
        if w1 is None:
            self.w1 = np.random.randn(4,3)
        else:
            self.w1 = w1
        if w2 is None:
            self.w2 = np.random.randn(1,4)
        else:
            self.w2 = w2
        if b1 is None:
            self.b1 = np.random.randn(4,1)
        else:
            self.b1 = b1
        if b2 is None:
            self.b2 = np.random.randn(1,1)
        else:
            self.b2 = b2
            
    def network(self, x):
        
        def sigmoid(z):
            return 1/(1+np.exp(-z))

        def layer(w, x, b): 
            z = np.dot(w, x) + b
            return sigmoid(z)
        
        a = layer(self.w1, x, self.b1)
        yhat = layer(self.w2, a, self.b2)
        return yhat
    
    def calc_cost(self, x, y):
        y_hat =  self.network(x)
        return RMSE(y, y_hat)
    
    def reproduce(self, other_ind, children=2):
        
        def mutate(char):
            tomutate = np.random.choice([True, False], char.shape, p =[0.1, 0.9])
            mutations =  np.random.randn(*char.shape)
            char[tomutate] = mutations[tomutate]
            return char

        def get_child():
            w1_prob = np.random.randint(0,2, other_ind.w1.shape)
            child_w1 = np.select([w1_prob==0, w1_prob==1],[self.w1, other_ind.w1])
            child_w1 = mutate(child_w1)
            w2_prob = np.random.randint(0,2, other_ind.w2.shape)
            child_w2 = np.select([w2_prob==0, w2_prob==1],[self.w2, other_ind.w2])
            b1_prob = np.random.randint(0,2, other_ind.b1.shape)
            child_b1 = np.select([b1_prob==0, b1_prob==1],[self.b1, other_ind.b1])
            b2_prob = np.random.randint(0,2, other_ind.b2.shape)
            child_b2 = np.select([b2_prob==0, b2_prob==1],[self.b2, other_ind.b2])
            ind = Individual(child_w1, child_w2, child_b1, child_b2)
            return ind
            
        kids = [get_child() for _ in range(children)]
        return kids

def get_top(pop, x, y, best=5):
    y_hats = [ind.network(x) for ind in pop]
    cost_of_pop = [RMSE(y, y_hat) for y_hat in y_hats]
    return np.take(pop, np.argsort(cost_of_pop)[:best])

def new_generation(best_indus):
    new_pop = []
    permus = list(permutations(best_indus, 2))
    for indu1, indu2 in permus:
        new_pop += indu1.reproduce(indu2, 5)
    return new_pop

def RMSE(y, y_hat):
    return np.sqrt((np.sum((y-y_hat)**2)/y.shape[0]))

def main(x, y, generations=5, best=5, start_population=5):
    population = [Individual() for _ in range(start_population)]
    for i in range(generations):
        best_induviduals = get_top(population, x, y, best=5)
        pop_mean = np.mean([bindu.calc_cost(x, y) for bindu in best_induviduals])
        print(f"Gen {i+1},  fitness RMSE is {pop_mean}")
        population = new_generation(best_induviduals) 
    score = best_induviduals[0].calc_cost(x,y)
    print(f"Finished generic optimization algorithm, best obtained model achieved a training RMSE of {score}")
    

if __name__ == "__main__":
    #read in the data
    df = pd.read_csv("neural_generic_algo_test_data.csv")
    x = df.drop("y", axis=1).values
    x = np.rot90(x)
    y = df["y"].values

    main(x, y, generations=20)
