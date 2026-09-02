"""
train_spam_filter.py
Fine-tunes a pretrained ResNet18 as a binary relevant/irrelevant photo
classifier, using a fast "linear probe" approach: the pretrained backbone
(already trained on millions of ImageNet images) stays FROZEN, and only a
new final classification layer is trained. This is real transfer learning,
genuinely faster than full fine-tuning - practical for CPU training under
time pressure, at some cost to peak accuracy versus full fine-tuning.

Usage:
    python train_spam_filter.py \
        --dataset_dir ../data/spam_filter_dataset \
        --out spam_filter_model.pt \
        --epochs 5
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])

    dataset = ImageFolder(args.dataset_dir, transform=transform)
    print(f"Classes: {dataset.classes}")  # ['irrelevant', 'relevant'] alphabetical

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze the pretrained backbone - only the new final layer gets
    # gradients, which is what makes this fast enough for CPU training.
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} - train loss: {total_loss/len(train_loader):.4f} - val accuracy: {val_acc:.3f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes,
    }, args.out)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
