import numpy as np
import pandas as pd

def process_data(data):
    data[['UserId', 'ItemId']] = data['UserId:ItemId'].str.split(':', expand=True)
    data = data.drop('UserId:ItemId',axis=1)

    unique_users = set(data['UserId'])
    unique_items = set(data['ItemId'])

    users = {user_id: idx for idx, user_id in enumerate(unique_users)}
    items = {item_id: idx for idx, item_id in enumerate(unique_items)}

    
    return data,users,items

def _preprocess_data(X, train=True, verbose=True):
        """Maps user and item ids to their indexes.

        Parameters
        ----------
        X : pandas.DataFrame
            Dataset, must have 'u_id' for user ids, 'i_id' for item ids, and
            'rating' column names.
        train : boolean
            Whether or not X is the training set or the validation set.

        Returns
        -------
        X : numpy.array
            Mapped dataset.
        """
        print('Preprocessing data...\n')
        X = X.copy()

        if train:  # Mappings have to be created
            user_ids = X['UserId'].unique().tolist()
            item_ids = X['ItemId'].unique().tolist()

            n_users = len(user_ids)
            n_items = len(item_ids)

            user_idx = range(n_users)
            item_idx = range(n_items)

            user_mapping_ = dict(zip(user_ids, user_idx))
            item_mapping_ = dict(zip(item_ids, item_idx))

        X['UserId'] = X['UserId'].map(user_mapping_)
        X['ItemId'] = X['ItemId'].map(item_mapping_)

        # Tag validation set unknown users/items with -1 (enables
        # `fast_methods._compute_val_metrics` detecting them)
        X.fillna(-1, inplace=True)

        X['UserId'] = X['UserId'].astype(np.int32)
        X['ItemId'] = X['ItemId'].astype(np.int32)
        X = X[['UserId', 'ItemId', 'Rating']].values

        global_mean_ = np.mean(X[:, 2])

        return X


def main():
    data = pd.read_csv("ratings.csv")
    data,users,items = process_data(data)
    X = _preprocess_data(data)
    

if __name__ == "__main__":
    main()