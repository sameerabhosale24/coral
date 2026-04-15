import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os

# ========================= CONFIG =========================
data_dir = r'C:\Users\samee\OneDrive\Desktop\Coral_Classifier1\data'
batch_size = 16
num_epochs = 15
model_name = 'resnet18'         # you can change to 'vgg16' later
learning_rate = 0.001
num_workers = 2                 # safe value for Windows + RTX 4050
# =======================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device} → {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ====================== LOAD DATASET ======================
    train_path = os.path.join(data_dir, r'C:\Users\samee\OneDrive\Desktop\Coral_Classifier1\data\train')
    val_path   = os.path.join(data_dir, r'C:\Users\samee\OneDrive\Desktop\Coral_Classifier1\data\valid')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ Training folder not found at: {train_path}")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=val_path, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    num_classes = len(train_dataset.classes)
    print(f"✅ Found {num_classes} classes: {train_dataset.classes}")

    # ====================== LOAD MODEL ======================
    # Using weights=None to avoid any download/network issues for now
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("model_name must be 'resnet18' or 'vgg16'")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    # ====================== TRAINING ======================
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total

        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}_coral.pth')
            print(f"✅ Best model saved (Val Acc: {val_acc:.2f}%)")

    print(f"\n🎉 Training completed! Best validation accuracy: {best_acc:.2f}%")