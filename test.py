# Testing the trained model
# import the saved model
import torch
# Loading the data
from dataloader import  val_loader

# Building a model 
from model_loader import modelling, loadModel

# Validating the accuracy
def testAccuracy(validate_path, model_name):

    val_load = val_loader(validate_path)
    model = modelling(model_name)
    validation_loss = []
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for images, labels in val_load:

            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)


