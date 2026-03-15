import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import cv2

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

def preprocess_frame(frame, image_size=224):
    """预处理摄像头帧"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 转换为PIL图像
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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

def process_camera(model, device='cuda'):
    """从摄像头实时检测表情"""
    # 尝试打开摄像头
    print("尝试打开摄像头...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 使用DirectShow后端
    
    if not cap.isOpened():
        print("无法打开摄像头，请检查摄像头是否被其他程序占用")
        return
    
    print("摄像头已打开，按 'q' 键退出")
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        try:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧，重试...")
                import time
                time.sleep(0.1)
                continue
            
            # 预处理帧
            image_tensor, _ = preprocess_frame(frame)
            
            # 预测表情
            predicted_idx, confidence, probabilities = predict_emotion(model, image_tensor, device)
            
            # 获取预测结果
            predicted_emotion = emotion_labels[predicted_idx]
            predicted_emotion_cn = emotion_labels_cn[predicted_emotion]
            emoji = emotion_emojis[predicted_emotion]
            
            # 打印详细的预测信息
            print(f"预测结果: {emoji} {predicted_emotion_cn} (置信度: {confidence:.2f})")
            print("所有表情的概率:")
            for i, (emotion, prob) in enumerate(zip(emotion_labels, probabilities)):
                emotion_cn = emotion_labels_cn[emotion]
                emoji = emotion_emojis[emotion]
                print(f"  {emoji} {emotion_cn}: {prob:.3f}")
            
            # 在帧上绘制结果
            # 使用更简单的显示方式，避免中文和emoji乱码
            emotion_display = predicted_emotion  # 使用英文标签
            cv2.putText(frame, f'{emotion_display}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Confidence: {confidence:.2f}', (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 显示帧
            cv2.imshow('表情识别', frame)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            import time
            time.sleep(0.1)
            continue
    
    # 释放摄像头
    try:
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"释放资源时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='表情识别摄像头实时检测脚本')
    parser.add_argument('--model', type=str, default='emotion_model_best.pth',
                       help='模型权重文件路径')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cuda 或 cpu)')
    
    args = parser.parse_args()
    
    # 检查参数
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
    
    # 从摄像头实时检测表情
    process_camera(model, args.device)

if __name__ == '__main__':
    main()