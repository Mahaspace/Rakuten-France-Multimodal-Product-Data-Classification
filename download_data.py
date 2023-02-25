import pandas as pd
import numpy as np
import os
import requests
from sklearn.model_selection import train_test_split


import os

# file source:
# https://www.kaggle.com/datasets/moussasacko/rakuten-france-multimodal-product-classification?resource=download
url = ["https://drive.google.com/uc?export=download&id=1gUMr9ltyFB5P2fHsM-tlJfO3PWMxx25k"
, "https://drive.google.com/uc?export=download&id=1Y3XLmowrwstA4OCgsvopnhuSDJBRBwH1"]



public_path = "./data/public/"
private_path = "./data/"

file_names = ['X_train.csv', 'Y_train.csv'] 
file_path = [os.path.join(private_path, file_name) for file_name in file_names]

try:
    os.stat(private_path)
except OSError:
    os.mkdir(private_path)

try:
    os.stat(public_path)
except OSError:
    os.mkdir(public_path)
    
print("Downloading the data files. This could take a couple of minutes...")   
for i in range(len(url)):
# Send an HTTP request to the download link
    r = requests.get(url[i])
# Save the response content to a file
    open(file_path[i], 'wb').write(r.content)
print("Files downloaded...")



X = pd.read_csv(os.path.join('data','X_train.csv'))
Y = pd.read_csv(os.path.join('data','Y_train.csv'))
# 2 - Perform any data cleaning and split into private train/test subsets
def split_data(X, Y, test_size=0.2, random_state=57): 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test

# specify the random_state to ensure reproducibility
X_train, X_test, Y_train, Y_test = split_data(X, Y)

def join_data(X, Y):
     return pd.concat([X, Y], axis=1)
    
df_public_train = join_data(X_train, Y_train)
df_public_test = join_data(X_test, Y_test)

df_public_train.to_csv(os.path.join('data', 'public','train.csv'), index=False)
df_public_test.to_csv(os.path.join('data','public',  'test.csv'), index=False)
