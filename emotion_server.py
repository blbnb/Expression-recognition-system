import socket
import threading
import pickle
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

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

def preprocess_image(image_data, image_size=224):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 从字节数据创建图像
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    # 应用预处理
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    return image_tensor

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
    
    # 获取预测结果
    predicted_emotion = emotion_labels[predicted_idx]
    predicted_emotion_cn = emotion_labels_cn[predicted_emotion]
    emoji = emotion_emojis[predicted_emotion]
    
    return {
        'emotion': predicted_emotion,
        'emotion_cn': predicted_emotion_cn,
        'emoji': emoji,
        'confidence': confidence
    }

def handle_client(client_socket, model, device):
    """处理客户端连接"""
    try:
        while True:
            # 接收图像数据
            data = client_socket.recv(1024 * 1024)  # 1MB缓冲区
            if not data:
                break
            
            # 反序列化图像数据
            image_data = pickle.loads(data)
            
            # 预处理图像
            image_tensor = preprocess_image(image_data)
            
            # 预测表情
            result = predict_emotion(model, image_tensor, device)
            
            # 序列化结果并发送回客户端
            client_socket.send(pickle.dumps(result))
            
    except Exception as e:
        print(f"处理客户端时出错: {str(e)}")
    finally:
        client_socket.close()

def start_server(host='0.0.0.0', port=5000, model_path='emotion_model_best.pth', device='cuda'):
    """启动服务器"""
    # 加载模型
    model = load_model(model_path, device)
    
    # 创建服务器套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    
    print(f"服务器已启动，监听 {host}:{port}")
    print(f"使用设备: {device}")
    
    while True:
        # 接受客户端连接
        client_socket, addr = server_socket.accept()
        print(f"客户端连接: {addr}")
        
        # 创建新线程处理客户端
        client_thread = threading.Thread(
            target=handle_client, 
            args=(client_socket, model, device)
        )
        client_thread.daemon = True
        client_thread.start()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='表情识别服务器')
    parser.add_argument('--model', type=str, default='emotion_model_best.pth',
                       help='模型权重文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda 或 cpu)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务器主机')
    parser.add_argument('--port', type=int, default=5000,
                       help='服务器端口')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    start_server(args.host, args.port, args.model, args.device)