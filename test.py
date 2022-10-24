import torch
import os
from torch import nn

# Loading the data
from dataloader import  val_loader

# Building a model 
from model_loader import modelling, loadModel

# Validating the accuracy
def testAccuracy(validate_path, model_name):

    val_load = val_loader(validate_path)
    validation_loss = 0
    criterion = nn.NLLLoss()

    if os.path.exists(f'{model_name}.pth'):
      model = loadModel(model_name)
    else:
      model = modelling(model_name)

    accuracy = 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
            model.eval()
            for images, labels in val_load:
                images, labels = images.to(device), labels.to(device)
                log_ps = model.forward(images)
                validation_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

    
    # compute the accuracy over all test images
    accuracy = (accuracy / len(val_load))
    val_loss = validation_loss / len(val_load)
    return accuracy, val_loss


