import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

MODEL_PATH = 'flower_model_complete.pth'
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return CLASS_NAMES[predicted.item()], round(confidence.item() * 100, 2)

if __name__ == '__main__':
    test_images = [
        'test1.jpg',
        'test2.jpg',
        'test3.jpg',
        'test4.jpg'
    ]

    print("Batch Inference Results:")
    for i, img_path in enumerate(test_images):
        if os.path.exists(img_path):
            label, conf = predict(img_path)
            print(f"Sample {i+1}: {img_path} → {label} ({conf}%)")
        else:
            print(f"Sample {i+1}: {img_path} → FILE NOT FOUND")
