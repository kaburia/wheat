from torchvision import models
import torch
from torch import nn, optim

# Creating a model
# Check for the layers required(input) in the Linear dense network
# An MLP Network

# prettrained networks
densenet121 = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')
alexnet = models.alexnet(weights='AlexNet_Weights.DEFAULT')


# Models to choose from
model2 = {'densenet': densenet121,
          'resnet': resnet18,
          'alexnet': alexnet}


def modelling(model_name):
    # Use a pretrained model
    model = model2[model_name.lower()]
    
    # Freeze parameters to avoid backpropagation through them
    for param in model.parameters():
        param.requires_grad = False

    if model_name.lower() == 'alexnet':
        model.classifier = nn.Sequential(nn.Linear(9216, 4896),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(4896, 2448),
                                        nn.ReLU(),
                                        nn.Linear(2448,102),
                                        nn.LogSoftmax(dim=1))
    elif model_name.lower() == 'densenet':
        model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256,102),
                                    nn.LogSoftmax(dim=1))
    elif model_name.lower() == 'resnet':
        model.fc = nn.Sequential(nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256,102),
                                 nn.LogSoftmax(dim=1))

    return model

# Function to save the model
def saveModel(model_name):
    model = modelling(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    torch.save(model.state_dict(), f'{model_name}.pth')

# Load the model
def loadModel(model_name):
    model = modelling(model_name)
    checkpoint = torch.load(f'{model_name}.pth', map_location=torch.device('cpu'))
    return model.load_state_dict(checkpoint)