import torch
from torch import nn
import pickle
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os

# Import the Net class
from app.net import Net

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Net':
            from app.net import Net
            return Net
        return super().find_class(module, name)

def load_model():
    best_model_config = {
        'l1': 256,  # example value, use the actual value from best_trial.config
        'l2': 512   # example value, use the actual value from best_trial.config
    }

    # Initialize the model
    model = Net(best_model_config['l1'], best_model_config['l2'])

    # Load the model state
    current_working_directory = os.getcwd()
    print(f"This is: {current_working_directory}")
    model_filename = os.path.join(current_working_directory, 'app', 'best_model.pkl')
    print(f"Model file path: {model_filename}")

    with open(model_filename, 'rb') as f:
        model = CustomUnpickler(f).load()

    print("Model loaded successfully")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    return model, device

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def predict_image_class(model, device, image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()






