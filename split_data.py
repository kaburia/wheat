# split the dataset to train, validate and test 
# Based on the 4 labels (healthy, septoria, brown_rust, yellow_rust)

import shutil
from os import listdir
import os
from random import sample



# Building and splitting the dataset
def split_data(path):
    labels = [label for label in listdir(path) if os.path.isdir(label)]
    print(labels)
    method = ['Train','Validate', 'Test']
    for meth in method:
        for label in labels:
             # creating a  folder if they do not exist
            
                if os.path.exists(f'{meth}') is False:
                    os.mkdir(f'{meth}')
                    if os.path.exists(f'{meth}/{label}') is False:
                        os.mkdir(f'{meth}/{label}')
                elif os.path.exists(f'{meth}') is True and os.path.exists(f'{meth}/{label}') is False:
                    os.mkdir(f'{meth}/{label}')

                if meth == 'Train':
                    length = int(len(listdir(path+label)) * 0.8) # choosing 80% of the values
                    sample_list = sample(listdir(path+label), length)
                    for file in sample_list:
                        shutil.copy(f'{path+label}/{file}', f'{meth}/{label}') 
                elif meth == 'Validate':
                    length = int(len(listdir(path+label)) * 0.5) # choosing 10% of the values
                    sample_list = sample(listdir(path+label), length)
                    for file in sample_list:
                        shutil.copy(f'{path+label}/{file}', f'{meth}/{label}') 
                else:
                    length = int(len(listdir(path+label))) # choosing the remaining values for test
                    sample_list = sample(listdir(path+label), length)
                    for file in sample_list:
                        shutil.copy(f'{path+label}/{file}', f'{meth}/{label}') 
    


split_data('YELLOW-RUST-19')
