# 1 hidden layer neural network with generic optimization algorithm
import numpy as np
import pandas as pd
import random
from itertools import permutations 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class Individual:
    def __init__(self, mutation_rate=0.1, w1=None, w2=None, b1=None, b2=None):
        self.mutation_rate = mutation_rate
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
        
        def relu(z):
            return np.maximum(0, z)

        def layer(w, x, b): 
            z = np.dot(w, x) + b
            return relu(z)
        
        a = layer(self.w1, x, self.b1)
        yhat = layer(self.w2, a, self.b2)
        return yhat
    
    def calc_cost(self, x, y):
        y_hat = self.predict(x) 
        return f1_score(y, y_hat, average="macro")
    
    def predict(self, x):
        return np.round(self.network(x)).flatten() ##round to 0 or 1 to get class
        
    def reproduce(self, other_ind, children=2):
        
        def mutate(char):
            mut=[self.mutation_rate, 1-self.mutation_rate]
            tomutate = np.random.choice([True, False], char.shape, p =mut)
            mutations =  np.random.randn(*char.shape)
            char[tomutate] = mutations[tomutate]
            return char

        def get_child():
            w1_prob = np.random.randint(0,2, other_ind.w1.shape)
            child_w1 = np.select([w1_prob==0, w1_prob==1],[self.w1, other_ind.w1])
            child_w1 = mutate(child_w1)
            w2_prob = np.random.randint(0,2, other_ind.w2.shape)
            child_w2 = np.select([w2_prob==0, w2_prob==1],[self.w2, other_ind.w2])
            child_w2 = mutate(child_w2)
            b1_prob = np.random.randint(0,2, other_ind.b1.shape)
            child_b1 = np.select([b1_prob==0, b1_prob==1],[self.b1, other_ind.b1])
            child_b1 = mutate(child_b1)
            b2_prob = np.random.randint(0,2, other_ind.b2.shape)
            child_b2 = np.select([b2_prob==0, b2_prob==1],[self.b2, other_ind.b2])
            child_b2 = mutate(child_b2)
            ind = Individual(self.mutation_rate, child_w1, child_w2, child_b1, child_b2)
            return ind
            
        kids = [get_child() for _ in range(children)]
        return kids
def get_top(pop, x, y, best=5):
    cost_of_pop = [1-ind.calc_cost(x, y) for ind in pop]
    return np.take(pop, np.argsort(cost_of_pop)[:best])

def new_generation(best_indus, add_parents):
    if add_parents:
        new_pop = [*best_indus]
    else:
        new_pop = []
    permus = list(permutations(best_indus, 2))
    for indu1, indu2 in permus:
        new_pop += indu1.reproduce(indu2, children=5)
    return new_pop

def main(X_train, y_train, X_test, y_test, generations=5, best=5, start_population=5, mutation_rate=0.1, add_parents=True):
    population = [Individual(mutation_rate) for _ in range(start_population)]
    for i in range(generations):
        best_induviduals = get_top(population, X_train, y_train, best=5)
        pop_mean_train = np.mean([bindu.calc_cost(X_train, y_train) for bindu in best_induviduals])
        pop_mean_test = np.mean([bindu.calc_cost(X_test, y_test) for bindu in best_induviduals])
        print(f"Gen {i+1}, fitness F1-score is {pop_mean_train} and a test F1-score of {pop_mean_test}")
        population = new_generation(best_induviduals, add_parents) 
    score_train = best_induviduals[0].calc_cost(X_train,y_train)
    score_test = best_induviduals[0].calc_cost(X_test,y_test)
    print(f"Finished generic optimization algorithm")
    print(f"Best obtained model achieved a training F1-score of {score_train}")
    print(f"Best obtained model achieved a test F1-score of {score_test}")
    

if __name__ == "__main__":
    #read in the data
    df = pd.read_csv("neural_generic_algo_test_data.csv")
    df_shuffles = shuffle(df, random_state=42) # always shuffle your data to avoid any biases that may emerge b/c of some order
    train, test = train_test_split(df_shuffles, test_size=0.3, random_state=42)

    x_train = train.drop("y", axis=1).values
    y_train = train["y"].values
    X_train = np.rot90(x_train)

    x_test = test.drop("y", axis=1).values
    y_test = test["y"].values
    X_test = np.rot90(x_test)

    main(X_train, y_train, X_test, y_test,
         generations=50, best=5,
         add_parents=True, mutation_rate=0.1)

