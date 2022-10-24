import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from dataloader import transforming
from model_loader import loadModel, modelling
# Visualizing the data

resnet = loadModel('densenet')
print(resnet)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transform = transforming()
    img = Image.open(image)
    return transform(img)

    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    return ax


''' 
Predict the class (or classes) of an image using a trained deep learning model.
'''
def predict(image_path, model, topk=3):
      
    image = process_image(image_path).unsqueeze(dim=0)

    with torch.no_grad():
      model.eval()
      output = model.forward(image)

    ps = torch.exp(output)
    ps, classes = ps.topk(topk, dim=1)
    return ps, classes



def view(image_path, model):
    
    probs, classes = predict(image_path, model) 
    image = process_image(image_path) 
      
    clas_im = [str(top_class) for top_class in classes.numpy()[0]]
    prbs = [pr for pr in probs.numpy()[0]]
    x = plt.barh(clas_im, prbs, color='purple')
    image = imshow(image)
    return plt.show()


view('Train\Brown_rust\Brown_rust002.jpg', resnet)