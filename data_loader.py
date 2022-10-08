# split the dataset to train, validate and test 
# Based on the 4 labels (healthy, septoria, brown_rust, yellow_rust)
import glob
import shutil
from os import listdir
import os
from random import sample

labels = ['Brown_rust', 'Healthy', 'septoria', 'Yellow_rust']

# Building a training dataset
def train(labels):
    for label in labels:
        length = int(len(listdir(label)) * 0.8) # choosing 80% of the values
        sample_list = sample(listdir(label), length)
        # for file in sample_list:
        # creating a training/label folder if they do not exist
        if os.path.exists('Train') is False:
                os.mkdir('Train')
                if os.path.exists(f'Train/{label}') is False:
                    os.mkdir(f'Train/{label}')
        else:
                os.mkdir(f'Train/{label}')
                
                


# print(os.listdir())
train(labels)
