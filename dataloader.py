import torch
from torchvision import transforms, datasets

# Loading the split datasets
# Transforming and augmenting
def trainloader(trainpath):
    train_transform = transforms.Compose([transforms.RandomRotation(10),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], # Mean
                                                                    [0.229, 0.224, 0.225])])# Standard deviation

    train_dataset = datasets.ImageFolder(trainpath, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader

# validation and test transforms
def transforming():
    test_transforms  = transforms.Compose((transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])))
    return test_transforms

# Train validation data
def val_loader(validate_path):
    test_transforms  = transforming()
    val_dataset = datasets.ImageFolder(validate_path, transform=test_transforms)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64)
    return valloader


def testloader(test_path):
    test_transforms  = transforming()
    test_dataset = datasets.ImageFolder(test_path, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    return test_loader
