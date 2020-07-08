import torch
from PIL import Image
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg11(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

model = load_checkpoint('checkpoint.pth')

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

### Args ###

ap = argparse.ArgumentParser()

ap.add_argument('--checkpoint', '--ckp', required=True, help='checkpoint filename')
ap.add_argument('--topk', '--k', type=int, required=True, help='return top K most likely classes')
ap.add_argument('--category_names', '--cat', type=str, help='use mapping of categories to real names')
ap.add_argument('--gpu', type=str, help='use GPU for training')
ap.add_argument('--image_path', type=str, required=True, help='path of image to be predicted')

args = ap.parse_args()
