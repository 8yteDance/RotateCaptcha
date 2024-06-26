import os
import random
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 360)  # 360 classes for 0-358 degrees rotation

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('rotate_model.pth'))
model.eval()

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
])

# Function to predict rotation angle
def predict_rotation_angle(image_path, model, transform):
    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_angle = predicted.item()  # Get the predicted angle
    return predicted_angle

if __name__ == '__main__':
    
    for name in random.sample(os.listdir("360"), 10):

        image_path = f'360/{name}'
        angle = image_path.split("_")[2].split(".")[0]
        
        # Predict rotation angle
        ts = time.time()
        predicted_angle = predict_rotation_angle(image_path, model, transform)
        print("路径：", image_path,", 真实角度：" , angle,", 预测角度:", predicted_angle, "耗时：",time.time()-ts)
    
