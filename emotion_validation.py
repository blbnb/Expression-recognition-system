import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc
)
import seaborn as sns
import argparse
from tqdm import tqdm
import pandas as pd

# 表情类别
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 表情对应的中文
emotion_labels_cn = {
    'angry': '愤怒',
    'disgust': '厌恶',
    'fear': '恐惧',
    'happy': '开心',
    'neutral': '中性',
    'sad': '悲伤',
    'surprise': '惊讶'
}

# 表情对应的emoji
emotion_emojis = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

# 自定义数据集类
class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
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
        
        return image, label, img_path

# 定义模型
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=len(emotion_labels)):
        super(EmotionClassifier, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def load_model(model_path, device='cuda'):
    """加载训练好的模型"""
    model = EmotionClassifier().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"模型已加载: {model_path}")
    return model

def evaluate_model(model, test_loader, criterion, device):
    """评估模型性能"""
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_image_paths = []
    
    with torch.no_grad():
        for images, labels, image_paths in tqdm(test_loader, desc='评估中'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_image_paths.extend(image_paths)
    
    avg_loss = test_loss / len(test_loader)
    
    return avg_loss, np.array(all_labels), np.array(all_predictions), np.array(all_probabilities), all_image_paths

def plot_confusion_matrix(cm, save_path='confusion_matrix_validation.png'):
    """绘制混淆矩阵"""
    plt.figure(figsize=(12, 10))
    
    # 使用中文标签
    labels_cn = [f'{emotion_emojis[label]} {emotion_labels_cn[label]}' for label in emotion_labels]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_cn, yticklabels=labels_cn,
                cbar_kws={'label': '样本数量'})
    
    plt.title('混淆矩阵', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存: {save_path}")
    plt.close()

def plot_normalized_confusion_matrix(cm, save_path='confusion_matrix_normalized.png'):
    """绘制归一化的混淆矩阵"""
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    
    labels_cn = [f'{emotion_emojis[label]} {emotion_labels_cn[label]}' for label in emotion_labels]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=labels_cn, yticklabels=labels_cn,
                cbar_kws={'label': '比例'})
    
    plt.title('归一化混淆矩阵', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"归一化混淆矩阵已保存: {save_path}")
    plt.close()

def plot_class_metrics(metrics_df, save_path='class_metrics.png'):
    """绘制各类别性能指标"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    labels_cn = [f'{emotion_emojis[label]} {emotion_labels_cn[label]}' for label in emotion_labels]
    
    # 准确率
    axes[0, 0].bar(labels_cn, metrics_df['accuracy'], color='skyblue')
    axes[0, 0].set_title('各类别准确率', fontweight='bold')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 精确率
    axes[0, 1].bar(labels_cn, metrics_df['precision'], color='lightcoral')
    axes[0, 1].set_title('各类别精确率', fontweight='bold')
    axes[0, 1].set_ylabel('精确率')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 召回率
    axes[1, 0].bar(labels_cn, metrics_df['recall'], color='lightgreen')
    axes[1, 0].set_title('各类别召回率', fontweight='bold')
    axes[1, 0].set_ylabel('召回率')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # F1分数
    axes[1, 1].bar(labels_cn, metrics_df['f1-score'], color='gold')
    axes[1, 1].set_title('各类别F1分数', fontweight='bold')
    axes[1, 1].set_ylabel('F1分数')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"类别性能指标图已保存: {save_path}")
    plt.close()

def plot_per_class_accuracy(cm, save_path='per_class_accuracy.png'):
    """绘制每个类别的准确率"""
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    
    labels_cn = [f'{emotion_emojis[label]} {emotion_labels_cn[label]}' for label in emotion_labels]
    colors = plt.cm.Set3(np.linspace(0, 1, len(emotion_labels)))
    
    bars = plt.bar(labels_cn, per_class_acc, color=colors)
    
    # 在柱状图上添加数值
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('各类别准确率', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('准确率', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"各类别准确率图已保存: {save_path}")
    plt.close()

def generate_detailed_report(labels, predictions, probabilities, output_dir='validation_results'):
    """生成详细的验证报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 总体准确率
    overall_accuracy = accuracy_score(labels, predictions)
    print(f"\n{'='*60}")
    print(f"总体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"{'='*60}\n")
    
    # 分类报告
    report = classification_report(labels, predictions, target_names=emotion_labels, output_dict=True)
    print("详细分类报告:")
    print(classification_report(labels, predictions, target_names=emotion_labels))
    
    # 保存分类报告到文件
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"表情识别验证报告\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"总体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n\n")
        f.write("详细分类报告:\n")
        f.write(classification_report(labels, predictions, target_names=emotion_labels))
    print(f"分类报告已保存: {report_path}")
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, os.path.join(output_dir, 'confusion_matrix.png'))
    plot_normalized_confusion_matrix(cm, os.path.join(output_dir, 'confusion_matrix_normalized.png'))
    
    # 各类别性能指标
    metrics_df = pd.DataFrame({
        'emotion': emotion_labels,
        'emotion_cn': [emotion_labels_cn[label] for label in emotion_labels],
        'accuracy': [report[label]['precision'] for label in emotion_labels],
        'precision': [report[label]['precision'] for label in emotion_labels],
        'recall': [report[label]['recall'] for label in emotion_labels],
        'f1-score': [report[label]['f1-score'] for label in emotion_labels],
        'support': [report[label]['support'] for label in emotion_labels]
    })
    
    # 保存指标到CSV
    metrics_path = os.path.join(output_dir, 'class_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"类别指标已保存: {metrics_path}")
    
    # 绘制图表
    plot_class_metrics(metrics_df, os.path.join(output_dir, 'class_metrics.png'))
    plot_per_class_accuracy(cm, os.path.join(output_dir, 'per_class_accuracy.png'))
    
    return overall_accuracy, metrics_df

def analyze_errors(labels, predictions, probabilities, image_paths, output_dir='validation_results'):
    """分析错误预测的样本"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 找出错误预测的样本
    incorrect_mask = labels != predictions
    incorrect_indices = np.where(incorrect_mask)[0]
    
    print(f"\n错误分析:")
    print(f"总样本数: {len(labels)}")
    print(f"错误预测数: {len(incorrect_indices)}")
    print(f"错误率: {len(incorrect_indices)/len(labels)*100:.2f}%")
    
    # 按真实表情分组统计错误
    error_analysis = {}
    for idx in incorrect_indices:
        true_label = emotion_labels[labels[idx]]
        predicted_label = emotion_labels[predictions[idx]]
        confidence = probabilities[idx][predictions[idx]]
        
        if true_label not in error_analysis:
            error_analysis[true_label] = []
        
        error_analysis[true_label].append({
            'predicted': predicted_label,
            'confidence': confidence,
            'image_path': image_paths[idx]
        })
    
    # 保存错误分析
    error_path = os.path.join(output_dir, 'error_analysis.txt')
    with open(error_path, 'w', encoding='utf-8') as f:
        f.write("错误预测分析\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"总样本数: {len(labels)}\n")
        f.write(f"错误预测数: {len(incorrect_indices)}\n")
        f.write(f"错误率: {len(incorrect_indices)/len(labels)*100:.2f}%\n\n")
        
        for true_label, errors in error_analysis.items():
            f.write(f"\n真实表情: {emotion_emojis[true_label]} {emotion_labels_cn[true_label]} ({true_label})\n")
            f.write(f"错误数量: {len(errors)}\n")
            
            # 统计最常见的错误预测
            predicted_counts = {}
            for error in errors:
                pred = error['predicted']
                if pred not in predicted_counts:
                    predicted_counts[pred] = 0
                predicted_counts[pred] += 1
            
            f.write("最常见的错误预测:\n")
            for pred, count in sorted(predicted_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  - {emotion_emojis[pred]} {emotion_labels_cn[pred]} ({pred}): {count}次\n")
            
            f.write("\n")
    
    print(f"错误分析已保存: {error_path}")

def main():
    parser = argparse.ArgumentParser(description='表情识别模型验证脚本')
    parser.add_argument('--model', type=str, default='emotion_model_best.pth',
                       help='模型权重文件路径')
    parser.add_argument('--data_dir', type=str, default='kagglehub/datasets/msambare/fer2013/versions/1/test',
                       help='测试数据目录')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--output_dir', type=str, default='validation_results',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda 或 cpu)')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件 {args.model} 不存在")
        return
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录 {args.data_dir} 不存在")
        return
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    print(f"使用设备: {args.device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    test_dataset = EmotionDataset(args.data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 加载模型
    model = load_model(args.model, args.device)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 评估模型
    print("\n开始评估模型...")
    test_loss, labels, predictions, probabilities, image_paths = evaluate_model(
        model, test_loader, criterion, args.device
    )
    
    print(f"\n测试损失: {test_loss:.4f}")
    
    # 生成详细报告
    overall_accuracy, metrics_df = generate_detailed_report(
        labels, predictions, probabilities, args.output_dir
    )
    
    # 错误分析
    analyze_errors(labels, predictions, probabilities, image_paths, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"验证完成！结果已保存到: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
