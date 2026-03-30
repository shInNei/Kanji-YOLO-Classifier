import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import pickle

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class ResNetTrainer:
    def __init__(self, model, train_loader, val_loader, device, checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.best_acc = 0.0
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch [{epoch}]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/total:.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return running_loss / len(self.train_loader), 100. * correct / total

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Val   Epoch [{epoch}]")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # THÊM DÒNG NÀY ĐỂ GIẢI PHÓNG CACHE GPU LIÊN TỤC
                torch.cuda.empty_cache() 
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/total:.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        val_acc = 100. * correct / total
        self.save_best_model(val_acc)
        return running_loss / len(self.val_loader), val_acc

    def save_best_model(self, current_acc):
        if current_acc > self.best_acc:
            self.best_acc = current_acc
            path = os.path.join(self.checkpoint_dir, 'best_resnet_model.pth')
            torch.save(self.model.state_dict(), path)
            print(f"--> Best model saved with Accuracy: {self.best_acc:.2f}%")

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            self.train_one_epoch(epoch)
            self.validate(epoch)

if __name__ == '__main__':
    from dataLoader import DataLoaderManager
    
    manager = DataLoaderManager(db_path='data_warehouse.db', lake_dir='Data_Lake')
    train_loader, val_loader, test_loader, encoder = manager.create_dataloaders()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(num_classes=len(encoder.classes_))
    
    trainer = ResNetTrainer(model, train_loader, val_loader, device)
    trainer.fit(epochs=10)