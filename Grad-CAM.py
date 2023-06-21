import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from train import *
from RANet import *

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

    def get_feature_maps(self, x):
        self.feature_maps = []
        self.gradients = []
        for name, module in self.model.named_modules():
            x = module(x)
            if name == self.target_layer:
                self.feature_maps.append(x)
                x.register_hook(self.save_gradient)
        return x

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradcam(self, x, class_idx):
        self.model.zero_grad()
        output = self.model(x)
        output[:, class_idx].backward()
        weights = torch.mean(self.gradients[0], axis=(2, 3)).unsqueeze(2).unsqueeze(3)
        gradcam = torch.sum(weights * self.feature_maps[-1], axis=1).squeeze()
        gradcam = F.relu(gradcam)
        gradcam = F.interpolate(gradcam.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        gradcam = gradcam.squeeze().numpy()
        gradcam = np.maximum(gradcam, 0)
        gradcam = gradcam / np.max(gradcam)
        return gradcam


# Load model
model = models.ResNet(Bottleneck)
model.eval()

# Define target layer
target_layer = 'toplayer'

# Load image
img_path = 'F:/Datasets/dataset/WHU-RS19/Images/Bridge/bridge_08.jpg'
img = Image.open(img_path)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Apply transforms and convert to tensor
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)

# Create GradCAM object
gradcam = GradCAM(model, target_layer)

# Get predicted class
with torch.no_grad():
    output = model(img_tensor)
    class_idx = torch.argmax(output).item()

# Get GradCAM heatmap
heatmap = gradcam.get_gradcam(img_tensor, class_idx)

# Resize heatmap to match original image size
heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))

# Normalize heatmap values
heatmap = heatmap - np.min(heatmap)
heatmap = heatmap / np.max(heatmap)

# Apply heatmap to original image
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
result = cv2.addWeighted(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)

# Display result
cv2.imshow('GradCAM', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
