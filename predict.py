import torch
from PIL import Image
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from train.py import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    model.cuda()

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    with open(path, 'rb') as f:
        with Image.open(f) as image:
            return image.convert('RGB')

print(test_transforms(pil_loader(test_dir + "/1/image_06743.jpg")))

def predict(image, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
    image_tensor = pil_loader(image)
    image_tensor = test_transforms(image_tensor).float()
    image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model.forward(input)
    output = torch.exp(output)
    with torch.no_grad():
        probs, classes = output.topk(topk, dim=1)
        idx_to_class = {v:k for k,v in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in classes[0].tolist()]
    return probs, classes
        
image = test_dir + '/1/image_06743.jpg'

probs, classes = predict(image, model)
print(probs)
print(classes)
