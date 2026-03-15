import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

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

# 定义模型（与训练时相同）
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
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model

def preprocess_image(image_path, image_size=224):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    # 应用预处理
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    return image_tensor, image

def predict_emotion(model, image_tensor, device='cuda'):
    """预测表情"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # 转换为numpy
    predicted_idx = predicted.item()
    confidence = confidence.item()
    probabilities = probabilities.cpu().numpy()[0]
    
    return predicted_idx, confidence, probabilities

def visualize_prediction(image, predicted_idx, confidence, probabilities, save_path=None):
    """可视化预测结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示图像
    ax1.imshow(image)
    predicted_emotion = emotion_labels[predicted_idx]
    predicted_emotion_cn = emotion_labels_cn[predicted_emotion]
    emoji = emotion_emojis[predicted_emotion]
    
    ax1.set_title(f'{emoji} {predicted_emotion_cn}\n({predicted_emotion})', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 显示置信度条形图
    y_pos = np.arange(len(emotion_labels))
    bars = ax2.barh(y_pos, probabilities, color='skyblue')
    
    # 高亮预测的表情
    bars[predicted_idx].set_color('coral')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'{emotion_emojis[label]} {emotion_labels_cn[label]}' for label in emotion_labels])
    ax2.set_xlabel('置信度', fontsize=12)
    ax2.set_title('表情预测概率分布', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    
    # 在条形图上添加数值
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def batch_predict(model, image_dir, device='cuda', output_dir=None):
    """批量预测目录中的所有图像"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    
    # 收集所有图像文件
    for ext in image_extensions:
        image_files.extend(Path(image_dir).rglob(f'*{ext}'))
        image_files.extend(Path(image_dir).rglob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"在目录 {image_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 预测每张图像
    results = []
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n处理图像 {idx}/{len(image_files)}: {image_file.name}")
        
        try:
            # 预处理图像
            image_tensor, image = preprocess_image(image_file)
            
            # 预测表情
            predicted_idx, confidence, probabilities = predict_emotion(model, image_tensor, device)
            
            # 获取预测结果
            predicted_emotion = emotion_labels[predicted_idx]
            predicted_emotion_cn = emotion_labels_cn[predicted_emotion]
            emoji = emotion_emojis[predicted_emotion]
            
            print(f"预测结果: {emoji} {predicted_emotion_cn} ({predicted_emotion})")
            print(f"置信度: {confidence:.3f}")
            
            # 显示前3个最可能的表情
            top3_indices = np.argsort(probabilities)[-3:][::-1]
            print("前3个可能的表情:")
            for i, idx in enumerate(top3_indices, 1):
                emotion = emotion_labels[idx]
                emotion_cn = emotion_labels_cn[emotion]
                emoji = emotion_emojis[emotion]
                prob = probabilities[idx]
                print(f"  {i}. {emoji} {emotion_cn} ({emotion}): {prob:.3f}")
            
            # 可视化
            if output_dir:
                save_path = os.path.join(output_dir, f"{image_file.stem}_prediction.png")
            else:
                save_path = None
            
            visualize_prediction(image, predicted_idx, confidence, probabilities, save_path)
            
            # 保存结果
            results.append({
                'image': str(image_file),
                'predicted_emotion': predicted_emotion,
                'predicted_emotion_cn': predicted_emotion_cn,
                'confidence': confidence,
                'probabilities': probabilities.tolist()
            })
            
        except Exception as e:
            print(f"处理图像 {image_file} 时出错: {str(e)}")
    
    # 保存批量预测结果
    if output_dir:
        import json
        results_path = os.path.join(output_dir, 'batch_predictions.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n批量预测结果已保存到: {results_path}")

def main():
    parser = argparse.ArgumentParser(description='表情识别推理脚本')
    parser.add_argument('--model', type=str, default='emotion_model_best.pth',
                       help='模型权重文件路径')
    parser.add_argument('--image', type=str, default=None,
                       help='单张图像路径')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='图像目录路径（批量预测）')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda 或 cpu)')
    
    args = parser.parse_args()
    
    # 检查参数
    if args.image is None and args.image_dir is None:
        print("错误: 必须指定 --image 或 --image_dir")
        return
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件 {args.model} 不存在")
        return
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    print(f"使用设备: {args.device}")
    
    # 加载模型
    model = load_model(args.model, args.device)
    
    # 单张图像预测
    if args.image:
        if not os.path.exists(args.image):
            print(f"错误: 图像文件 {args.image} 不存在")
            return
        
        print(f"\n处理图像: {args.image}")
        
        # 预处理图像
        image_tensor, image = preprocess_image(args.image)
        
        # 预测表情
        predicted_idx, confidence, probabilities = predict_emotion(model, image_tensor, args.device)
        
        # 获取预测结果
        predicted_emotion = emotion_labels[predicted_idx]
        predicted_emotion_cn = emotion_labels_cn[predicted_emotion]
        emoji = emotion_emojis[predicted_emotion]
        
        print(f"\n预测结果: {emoji} {predicted_emotion_cn} ({predicted_emotion})")
        print(f"置信度: {confidence:.3f}")
        
        # 显示所有表情的概率
        print("\n所有表情的概率:")
        for i, (emotion, prob) in enumerate(zip(emotion_labels, probabilities)):
            emotion_cn = emotion_labels_cn[emotion]
            emoji = emotion_emojis[emotion]
            print(f"  {emoji} {emotion_cn} ({emotion}): {prob:.3f}")
        
        # 可视化
        output_path = os.path.join(args.output_dir, f"{Path(args.image).stem}_prediction.png") if args.output_dir else None
        visualize_prediction(image, predicted_idx, confidence, probabilities, output_path)
    
    # 批量预测
    elif args.image_dir:
        if not os.path.exists(args.image_dir):
            print(f"错误: 图像目录 {args.image_dir} 不存在")
            return
        
        batch_predict(model, args.image_dir, args.device, args.output_dir)

if __name__ == '__main__':
    main()
