# Required libraries
import torch
from torch import  nn
from torch.optim import Adam
import argparse

# Loading the data
from dataloader import trainloader, val_loader

# Building a model 
from model_loader import modelling

   
# Train a model
def train(img_path, model_name, epochs=1, learning_rate=0.03, device='cpu'):
    
    model = modelling(model_name)
    trainload = trainloader(img_path)
    val_load = val_loader(img_path)
    
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 5

    print(f'Beginning Training:\nDirectory: {img_path} Model: {model_name}, Epochs: {epochs}, Device: {device}')
    # epochs = 10
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainload:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_load:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(val_load):.3f}.. "
                    f"Test accuracy: {accuracy/len(val_load):.3f}")
                running_loss = 0
                model.train()

    # Saving the model with the model name
    torch.save(model.state_dict(), f'{model_name}.pth')


    

def parse_args():
    parser = argparse.ArgumentParser()
    # Add arguments for command line
    args = parser.parse_args()

    return args




def main():
    args = parse_args()

   

if __name__ == '__main__':
    main()



# Set directory to save checkpoints
# Choose a specific directory to perform Training on
# Create a function to obtain the path of the input directory
# Check the format of the images
# If scattered create labels and group to pass into ImageFolder easily

# Check the imports from data
