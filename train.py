# Required libraries
import torch
from torch import  nn
from torch.optim import Adam
import argparse
from time import time

# Loading the data
from dataloader import trainloader, val_loader

# Building a model 
from model_loader import modelling, saveModel

# Testing the accuracy
from test import testAccuracy

   
# Train a model
def train(trainpath, validate_path, model_name, epochs=1, learning_rate=0.03, device='cpu'):
    
    start = time()
    model = modelling(model_name)
    trainload = trainloader(trainpath)
    
    
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f'Beginning Training:\nDirectory: {trainpath} Model: {model_name}, Epochs: {epochs}, Device: {device}')

    train_losses = []
    train_loss_back = []
    best_accuracy = 0
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in trainload:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()# Zero the parameter gradients
            logps = model.forward(inputs)# Predict classes using images from the training set
            loss = criterion(logps, labels) # Compute the loss
            train_loss_back.append(loss.item()) # Loss before backpropagation to adjust the weights
            loss.backward()# Backpropagate the loss
            optimizer.step() 

            running_loss += loss.item()
            train_losses.append(running_loss) # Getting the loss at each run
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                validation = testAccuracy(validate_path, model_name)
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(val_load):.3f}.. "
                    f"Test accuracy: {accuracy/len(val_load):.3f}")
                running_loss = 0
                model.train()
                
    # Calculating the run time
    end = time()
    print(round(end-start, 4))
    # Saving the model with the model name
    # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
    accuracy = testAccuracy()
    print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
    
    # we want to save the model if the accuracy is the best
    if accuracy > best_accuracy:
        saveModel()
        best_accuracy = accuracy




    



# Set directory to save checkpoints
# Choose a specific directory to perform Training on
# Create a function to obtain the path of the input directory
# Check the format of the images
# If scattered create labels and group to pass into ImageFolder easily

# Check the imports from data
