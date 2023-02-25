import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split# 1 - read in or download the data.


X = pd.read_csv(os.path.join('data', 'X_train.csv'))
Y = pd.read_csv(os.path.join('data', 'Y_train.csv'))

# 2 - Perform any data cleaning and split into private train/test subsets
def split_data(X, Y, test_size=0.2, random_state=57):
    X_test, X_train, Y_test, Y_train = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test

# specify the random_state to ensure reproducibility
X_train, X_test, Y_train, Y_test = split_data(X, Y)
# write a fucntion that join the input and output dataframes
def join_data(X, Y):
     return pd.concat([X, Y], axis=1)
    
# 3 - Split public train/test subsets. In this case the private training
# data will be used as the public data
df_public_train = join_data(X_train, Y_train)
df_public_test = join_data(X_test, Y_test)

    
df_public_train.to_csv(os.path.join('data', 'public', 'train.csv'), index=False)
df_public_test.to_csv(os.path.join('data', 'public', 'test.csv'), index=False)