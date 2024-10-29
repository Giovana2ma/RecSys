import numpy as np
import pandas as pd
from process_data import *

class Funksvd:

    def __init__(self,X,users,items,n_factors,n_epochs,lr,reg):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.X = X
        self.users = users
        self.items = items
        self.num_users = len(users)
        self.num_items = len(items)
        self.global_mean = X['Rating'].mean()


    def shuffle(self):
        self.X = np.random.shuffle(self.X)
        return self.X

    def initialization(self):
        
        bu = np.zeros(self.num_users)
        bi = np.zeros(self.num_items)

        p = np.random.normal(0, .1, (self.num_users, self.n_factors))
        q = np.random.normal(0, .1, (self.num_items, self.n_factors))

        # p = 5 * np.random.rand(self.num_users, self.n_factors) / (5**0.5)
        # q = 5 * np.random.rand(self.num_items, self.n_factors) / (5**0.5)

        return bu, bi, p, q

    def run_iteration_(self,X, bu, bi, p, q):
        for i in range(2):
            user, item, rating = self.users[X['UserId'][i]],self.items[X['ItemId'][i]],X['Rating'][i]

            # Predict current rating
            pred = self.global_mean + bu[user] + bi[item]

            for factor in range(self.n_factors):
                pred += p[user, factor] * q[item, factor]

            err = rating - pred

            # Update biases
            bu[user] += self.lr * (err - self.reg * bu[user])
            bi[item] += self.lr * (err - self.reg * bi[item])

            # Update latent factors
            for factor in range(self.n_factors):
                puf = p[user][factor]
                qif = q[item, factor]

                p[user, factor] += self.lr * (err * qif - self.reg * puf)
                q[item, factor] += self.lr * (err * puf - self.reg * qif)

        return bu, bi, p, q
    
    def run_iteration(self,X, bu, bi, p, q):
    
        user_indices = np.array([self.users[uid] for uid in X['UserId']])
        item_indices = np.array([self.items[iid] for iid in X['ItemId']])
        ratings = X['Rating'].values


        # Predictions: global mean + user biases + item biases
        pred = self.global_mean + bu[user_indices] + bi[item_indices]
        pred += np.sum(p[user_indices] * q[item_indices], axis=1)

        # Calculate the error
        err = ratings - pred

        # Update biases
        bu[user_indices] += self.lr * (err - self.reg * bu[user_indices])
        bi[item_indices] += self.lr * (err - self.reg * bi[item_indices])

        # Update latent factors
        p[user_indices] += self.lr * (err[:, np.newaxis] * q[item_indices] - self.reg * p[user_indices])
        q[item_indices] += self.lr * (err[:, np.newaxis] * p[user_indices] - self.reg * q[item_indices])

        return bu, bi, p, q
    
    def run_iteration_mini_batch(self, X, bu, bi, p, q, batch_size):
        # Get user, item indices and ratings from X
        user_indices = np.array([self.users[uid] for uid in X['UserId']]) # (U,)
        item_indices = np.array([self.items[iid] for iid in X['ItemId']]) # (I,)
        ratings = X['Rating'].values


        # Iterate over mini-batches
        for start in range(0, len(X), batch_size):
            # Define the mini-batch range
            end = start + batch_size
            batch_user_indices = np.unique(user_indices[start:end])
            batch_item_indices = np.unique(item_indices[start:end])
            batch_ratings = ratings[start:end]
            

            # Calculate predictions for the batch
            pred  = p @ q.T
            pred  += self.global_mean 
            pred  += bu.reshape(-1,1)
            pred  += bi.reshape(1,-1)

            # U * F * I

            a = user_indices[start:end]
            b = item_indices[start:end]
            pred_in_batch = pred[a,b]

            # Calculate error
            err = batch_ratings - pred_in_batch

            # Update biases
            bu[batch_user_indices] += self.lr * (err - self.reg * bu[batch_user_indices])
            bi[batch_item_indices] += self.lr * (err - self.reg * bi[batch_item_indices])

            # Update latent factors
            p[batch_user_indices] += self.lr * (err[:, np.newaxis] * q[batch_item_indices] - self.reg * p[batch_user_indices])
            q[batch_item_indices] += self.lr * (err[:, np.newaxis] * p[batch_user_indices] - self.reg * q[batch_item_indices])

        return bu, bi, p, q


    def run_sgd(self):
        """Runs SGD algorithm, learning model weights.

        Parameters
        ----------
        X : numpy.array
            Training set, first column must be user indexes, second one item
            indexes, and third one ratings.
        X_val : numpy.array or None
            Validation set with the same structure as X.
        """

        bu, bi, p, q = self.initialization()

        # Run SGD
        for epoch_ix in range(self.n_epochs):
            print(epoch_ix)

            
            # X = self.shuffle()

            bu, bi, p, q = self.run_iteration_mini_batch(self.X, bu, bi, p, q,5)

            # if early_stopping
            #     break

        print(p,q)


def ratings():
    rate = pd.read_csv("ratings.csv")
    return rate


def main():
    rates = ratings()
    n_factors = 150
    n_epochs = 20
    lr = 0.005
    reg = 0.02
    X,num_users,num_items = process_data(rates[0:50])
    recomendation = Funksvd(X,num_users,num_items,n_factors,n_epochs,lr,reg)
    recomendation.run_sgd()

    

if __name__ == "__main__":
    main()
