import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import argparse
from tqdm import tqdm

# 表情类别
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def setup_distributed(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

# 自定义数据集类
class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 遍历所有类别文件夹
        for label_idx, label in enumerate(emotion_labels):
            label_dir = os.path.join(root_dir, label)
            if os.path.exists(label_dir):
                for img_name in os.listdir(label_dir):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(label_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 定义模型
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=len(emotion_labels)):
        super(EmotionClassifier, self).__init__()
        # 使用预训练的ResNet50模型（更大容量，适合多GPU）
        self.model = models.resnet50(pretrained=True)
        # 修改最后一层全连接层
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, rank):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm显示进度条（只在主进程显示）
    if rank == 0:
        train_loader = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 计算平均损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device, rank):
    """评估模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        # 使用tqdm显示进度条（只在主进程显示）
        if rank == 0:
            test_loader = tqdm(test_loader, desc='Evaluating')
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, all_labels, all_predictions

def plot_results(train_losses, train_accs, test_loss, test_accuracy, all_labels, all_predictions, rank):
    """绘制训练结果"""
    if rank != 0:
        return
    
    # 绘制训练损失和准确率曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, marker='o', linestyle='-', label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(epochs, train_accs, marker='o', linestyle='-', label='Train Accuracy', color='orange')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    print('Training history saved as training_history.png')
    
    # 生成分类报告
    print('\nClassification Report:')
    print(classification_report(all_labels, all_predictions, target_names=emotion_labels))
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png', dpi=300)
    print('Confusion matrix saved as confusion_matrix.png')

def main_worker(rank, world_size, args):
    """每个GPU的工作进程"""
    # 设置分布式环境
    setup_distributed(rank, world_size)
    
    # 数据集路径
    data_dir = 'kagglehub/datasets/msambare/fer2013/versions/1'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = EmotionDataset(train_dir, transform=train_transform)
    test_dataset = EmotionDataset(test_dir, transform=test_transform)
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    device = torch.device(f'cuda:{rank}')
    model = EmotionClassifier().to(device)
    
    # 包装模型为DistributedDataParallel
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 训练记录
    train_losses = []
    train_accs = []
    best_acc = 0.0
    
    # 训练循环
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # 确保每个epoch的采样顺序不同
        
        epoch_loss, epoch_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, rank)
        scheduler.step()
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        if rank == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # 每5个epoch评估一次
        if (epoch + 1) % 5 == 0:
            test_loss, test_accuracy, all_labels, all_predictions = evaluate(model, test_loader, criterion, device, rank)
            
            if rank == 0:
                print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
                
                # 保存最佳模型
                if test_accuracy > best_acc:
                    best_acc = test_accuracy
                    torch.save(model.module.state_dict(), 'emotion_model_best.pth')
                    print(f'Best model saved with accuracy: {best_acc:.4f}')
    
    # 最终评估
    test_loss, test_accuracy, all_labels, all_predictions = evaluate(model, test_loader, criterion, device, rank)
    
    if rank == 0:
        print(f'\nFinal Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        plot_results(train_losses, train_accs, test_loss, test_accuracy, all_labels, all_predictions, rank)
        
        # 保存最终模型
        torch.save(model.module.state_dict(), 'emotion_model_final.pth')
        print('Final model saved as emotion_model_final.pth')
    
    # 清理分布式环境
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='Emotion Recognition Training on Multi-GPU')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--world_size', type=int, default=4, help='Number of GPUs to use')
    args = parser.parse_args()
    
    # 检查可用的GPU数量
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    if available_gpus < args.world_size:
        print(f"Warning: Requested {args.world_size} GPUs but only {available_gpus} available")
        args.world_size = available_gpus
    
    # 打印GPU信息
    for i in range(available_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, {props.total_memory / 1e9:.2f} GB")
    
    # 使用多进程启动训练
    mp.spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)

if __name__ == '__main__':
    main()
