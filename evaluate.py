import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# 1. SETUP & PATHS
TEST_DIR = r'C:\Users\samee\OneDrive\Desktop\Coral_Classifier1\data\test' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing transform must match what was used in train_coral.py  
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test images
test_data = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
class_names = test_data.classes # ['Bleached', 'Healthy']

# 2. LOAD THE FINISHED MODEL
def load_resnet_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Load 15-epoch weights
    model_path = 'best_resnet18_coral.pth'
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found! Run main.py first.")
        exit()
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

# 3. EVALUATION LOOP
def run_evaluation():
    resnet = load_resnet_model()
    print(f"✅ ResNet-18 Loaded. Testing on {len(test_data)} images using {device}...")

    correct = 0
    total = 0
    
    # For a simple Confusion Matrix
    tp, fp, tn, fn = 0, 0, 0, 0 # True Pos, False Pos, True Neg, False Neg

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Analyzing Images"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = resnet(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Simple Confusion Matrix math (Assuming Healthy is 1, Bleached is 0)
            for p, l in zip(predicted, labels):
                if p == 1 and l == 1: tp += 1
                elif p == 1 and l == 0: fp += 1
                elif p == 0 and l == 0: tn += 1
                elif p == 0 and l == 1: fn += 1

    # 4. FINAL SCOREBOARD
    accuracy = (correct / total) * 100
    
    print("\n" + "="*45)
    print("      RESNET-18 EVALUATION REPORT")
    print("="*45)
    print(f"Overall Accuracy:  {accuracy:.2f}%")
    print(f"Total Images:      {total}")
    print(f"Correct:           {correct}")
    print(f"Incorrect:         {total - correct}")
    print("-" * 45)
    print("DETAIL BREAKDOWN (Confusion Matrix):")
    print(f"Successfully caught Healthy:  {tp}")
    print(f"Successfully caught Bleached: {tn}")
    print(f"Mistook Bleached as Healthy:  {fp} (False Positive)")
    print(f"Mistook Healthy as Bleached:  {fn} (False Negative)")
    print("="*45 + "\n")

if __name__ == '__main__':
    run_evaluation()