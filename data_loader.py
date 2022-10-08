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

        # creating a training/label folder
        # if os.listdir


print(os.listdir())
# train(labels)