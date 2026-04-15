import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    data_dir = r'C:\Users\samee\OneDrive\Desktop\Coral_Classifier1\data\test'
    batch_size = 16
    model_name = 'resnet18'

    # Same transform as validation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, r'C:\Users\samee\OneDrive\Desktop\Coral_Classifier1\data\test'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the best model
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(test_dataset.classes))
    else:
        raise ValueError("Only resnet18 supported for now")

    model.load_state_dict(torch.load(f'best_{model_name}_coral.pth', weights_only=True))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    print("Evaluating on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100 * correct / total
    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")

    # Detailed report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

import os

model_path = "best_resnet18_coral.pth"

if not os.path.exists(model_path):
    print("Model file not found!")
    print("Please download it from the link in README.md")
    exit()