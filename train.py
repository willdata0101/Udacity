# Imports here

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
trainset = datasets.ImageFolder(train_dir, transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle=True)

validset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
validloader = torch.utils.data.DataLoader(validset, batch_size = 64, shuffle=True)

testset = datasets.ImageFolder(test_dir, transform=test_transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle=True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    

model = models.vgg11(pretrained=True)

number_of_inputs = model.classifier[0].in_features

# Defining model with dropout added
for param in model.parameters():
    param.requires_grad = False
    
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('dropout', nn.Dropout(p=0.6)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 1000)),
                          ('relu', nn.ReLU()),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

if torch.cuda.is_available():
    model.cuda()
    
# Build and train the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

images, labels = next(iter(trainloader))

epochs = 1
steps = 0
print_every = 5
running_loss = 0
for epoch in range(epochs):
    for images, labels in trainloader:
        steps +=1
        
        images, labels = images.to(device), labels.to(device)
        
        #Training pass
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()
            valid_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    output = model.forward(images)
                    batch_loss = criterion(output, labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

# Do validation on the test set

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

images, labels = next(iter(trainloader))

epochs = 1
steps = 0
print_every = 5
running_loss = 0
for epoch in range(epochs):
    for images, labels in trainloader:
        steps +=1
        
        images, labels = images.to(device), labels.to(device)
        
        #Training pass
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    output = model.forward(images)
                    batch_loss = criterion(output, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
 
# TODO: Save the checkpoint
model.class_to_idx = trainset.class_to_idx

checkpoint = {'input_size': 25088, 'output_size': 102, 'epochs': 5, 
              'hidden_layer': 4096, 'learning_rate': 0.00001, 'class_to_idx': model.class_to_idx,
              'classifier': model.classifier,
              'optimizer': optimizer.state_dict,
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')

### Args ###

ap = argparse.ArgumentParser()

ap.add_argument('--data_dir', '--d', type=str, dest='data_dir', help='path to folder of flower images')
ap.add_argument('--save_dir', '--s', help='sets directory to save checkpoints')
ap.add_argument('--arch', '--a', default='vgg11', help='choose architecture')
ap.add_argument('--learning_rate', action='store', type=float, default='0.01', help='sets learning rate')
ap.add_argument('--hidden_units', action='store', type=int, default='512', help='sets hidden units')
ap.add_argument('--epochs', action='store', type=int, default='20', help='sets epochs')
ap.add_argument('--gpu', action='store_true', help='use GPU for training')

args = ap.parse_args()

print(args)
