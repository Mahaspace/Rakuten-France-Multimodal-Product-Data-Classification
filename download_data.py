import pandas as pd
import zipfile
import os
import requests
from sklearn.model_selection import train_test_split
from pathlib import Path


# file source:
# https://www.kaggle.com/datasets/moussasacko/rakuten-france-multimodal-product-classification?resource=download
URL = ["https://drive.google.com/uc?export=download&id=1gUMr9ltyFB5P2fHsM-tlJfO3PWMxx25k",
       "https://drive.google.com/uc?export=download&id=1Y3XLmowrwstA4OCgsvopnhuSDJBRBwH1",
       "https://drive.google.com/uc?export=download&id=1JcTua0Ze5QWPLQSn18Dvcv2pElnRHhme&confirm=t&uuid=ce3462a6-9335-43fe-b793-742763ea6a7a&at=ALgDtsxIxYEFNI9t5AfbVOqSK-J3:1677530325146"
       ]



PUBLIC_PATH = Path("./data/public/")
PRIVATE_PATH = Path("./data/")

FILE_NAMES = ['X_train.csv', 'Y_train.csv'] 
FILE_PATH = [os.path.join(PRIVATE_PATH, file_name) for file_name in FILE_NAMES]

PRIVATE_PATH.mkdir(parents=True, exist_ok=True)
PUBLIC_PATH.mkdir(parents=True, exist_ok=True)
    

def download_urls(url = URL, file_path = FILE_PATH):
    print("Downloading the data files. This could take a couple of minutes...")   
    for i in range(2):
        # Send an HTTP request to the download link
        r = requests.get(url[i])
        # Save the response content to a file
        open(file_path[i], 'wb').write(r.content)




def download_images(images_url  = URL[2], file_path = PRIVATE_PATH) :
    print("Downloading the images. this will take a while...")   
    r = requests.get(images_url)
    file_name = 'images.zip'
    file_path = os.path.join(PRIVATE_PATH, file_name) 
    # Save the response content to a file
    open(file_path, 'wb').write(r.content)
    return file_path


def unzip_images(file_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(PUBLIC_PATH)
    return  None




def split_data(X, Y, test_size=0.2, random_state=57): 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test


def join_data(X, Y):
     return pd.concat([X, Y.iloc[:,1]], axis=1)
    


if __name__ == "__main__":
    download_urls()
    file_path = download_images()
    unzip_images(file_path)
    
    
    X = pd.read_csv(os.path.join('data','X_train.csv'))
    Y = pd.read_csv(os.path.join('data','Y_train.csv'))
    
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    df_public_train = join_data(X_train, Y_train)
    df_public_test = join_data(X_test, Y_test)
    
    df_public_train.to_csv(os.path.join('data', 'public','train.csv'), index=False)
    df_public_test.to_csv(os.path.join('data','public',  'test.csv'), index=False)