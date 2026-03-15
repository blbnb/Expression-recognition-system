# 表情识别系统

## 项目简介

本项目是一个基于深度学习的表情识别系统，支持7种表情的识别，包括愤怒、厌恶、恐惧、开心、中性、悲伤和惊讶。系统提供了完整的训练、验证、推理和实时检测功能，可以部署在本地或服务器端。

## 项目结构

```
.
├── emotion_recognition_multi_gpu.py  # 多GPU训练脚本
├── emotion_validation.py              # 模型验证脚本
├── emotion_inference.py               # 推理预测脚本
├── emotion_camera.py                  # 摄像头实时检测脚本
├── emotion_test.py                    # 测试脚本
├── emotion_server.py                  # 服务器端脚本
├── emotion_client.py                  # 客户端脚本
├── requirements.txt                   # 依赖包
├── README.md                          # 使用说明
├── emotion_model_best.pth             # 最佳模型（验证集准确率最高）
├── emotion_model_final.pth            # 最终模型
├── training_history.png               # 训练损失和准确率曲线
├── confusion_matrix.png               # 混淆矩阵
└── training_output.log                # 训练输出日志
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (用于GPU训练，可选)
- OpenCV (用于摄像头访问)

## 模型训练

### 基础训练（使用4张GPU）

```bash
python emotion_recognition_multi_gpu.py
```

### 自定义参数训练

```bash
# 训练100个epoch
python emotion_recognition_multi_gpu.py --epochs 100

# 调整批次大小和学习率
python emotion_recognition_multi_gpu.py --batch_size 128 --lr 0.001

# 使用2张GPU
python emotion_recognition_multi_gpu.py --world_size 2
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 128 | 每GPU的批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--world_size` | 4 | 使用的GPU数量 |

### 训练输出

训练完成后会生成以下文件：

- `emotion_model_best.pth` - 最佳模型（验证集准确率最高）
- `emotion_model_final.pth` - 最终模型
- `training_history.png` - 训练损失和准确率曲线
- `confusion_matrix.png` - 混淆矩阵
- `training_output.log` - 训练输出日志

## 模型验证

### 在测试集上验证

```bash
# 使用默认参数
python emotion_validation.py

# 指定模型文件
python emotion_validation.py --model emotion_model_best.pth

# 调整批次大小
python emotion_validation.py --batch_size 128

# 使用CPU验证
python emotion_validation.py --device cpu
```

### 验证参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | emotion_model_best.pth | 模型权重文件路径 |
| `--data_dir` | kagglehub/datasets/msambare/fer2013/versions/1/test | 测试数据目录 |
| `--batch_size` | 64 | 批次大小 |
| `--output_dir` | validation_results | 输出目录 |
| `--device` | cuda | 设备 (cuda 或 cpu) |

### 验证输出

验证完成后会生成以下文件：

- `classification_report.txt` - 详细的分类报告
- `confusion_matrix.png` - 混淆矩阵
- `confusion_matrix_normalized.png` - 归一化混淆矩阵
- `class_metrics.csv` - 各类别性能指标
- `class_metrics.png` - 各类别性能指标图表
- `per_class_accuracy.png` - 各类别准确率图表
- `error_analysis.txt` - 错误预测分析

## 推理预测

### 单张图像预测

```bash
# 预测单张图像
python emotion_inference.py --image test_image.jpg

# 指定模型和输出目录
python emotion_inference.py --image test_image.jpg --model emotion_model_best.pth --output_dir predictions

# 使用CPU预测
python emotion_inference.py --image test_image.jpg --device cpu
```

### 批量图像预测

```bash
# 预测目录中的所有图像
python emotion_inference.py --image_dir ./test_images

# 指定输出目录
python emotion_inference.py --image_dir ./test_images --output_dir predictions
```

### 推理参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | emotion_model_best.pth | 模型权重文件路径 |
| `--image` | None | 单张图像路径 |
| `--image_dir` | None | 图像目录路径（批量预测） |
| `--output_dir` | predictions | 输出目录 |
| `--device` | cuda | 设备 (cuda 或 cpu) |

### 推理输出

- 单张预测：显示预测结果和置信度分布图
- 批量预测：为每张图像生成预测结果图，并生成 `batch_predictions.json` 文件

## 摄像头实时检测

### 运行实时检测

```bash
# 默认使用CPU运行
python emotion_camera.py

# 使用GPU运行（如果可用）
python emotion_camera.py --device cuda

# 指定模型文件
python emotion_camera.py --model emotion_model_best.pth
```

### 摄像头检测参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | emotion_model_best.pth | 模型权重文件路径 |
| `--device` | cpu | 设备 (cuda 或 cpu) |

### 操作说明

- 按 'q' 键退出程序
- 终端会显示详细的表情识别结果，包括所有表情的概率
- 摄像头画面会显示英文标签和置信度

## 客户端-服务器架构

### 启动服务器

```bash
# 在服务器端运行
python emotion_server.py --device cuda  # 如果服务器有GPU
# 或
python emotion_server.py --device cpu   # 如果服务器只有CPU

# 指定主机和端口
python emotion_server.py --host 0.0.0.0 --port 5000
```

### 运行客户端

```bash
# 在客户端（轻薄本）运行
python emotion_client.py

# 指定服务器IP和端口
python emotion_client.py --host 服务器IP地址 --port 5000
```

### 服务器参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | emotion_model_best.pth | 模型权重文件路径 |
| `--device` | cuda | 设备 (cuda 或 cpu) |
| `--host` | 0.0.0.0 | 服务器主机 |
| `--port` | 5000 | 服务器端口 |

### 客户端参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | localhost | 服务器主机 |
| `--port` | 5000 | 服务器端口 |

## 表情类别

模型支持7种表情识别：

| 英文 | 中文 | Emoji |
|------|------|-------|
| angry | 愤怒 | 😠 |
| disgust | 厌恶 | 🤢 |
| fear | 恐惧 | 😨 |
| happy | 开心 | 😊 |
| neutral | 中性 | 😐 |
| sad | 悲伤 | 😢 |
| surprise | 惊讶 | 😲 |

## 性能指标

### 预期性能（在FER2013数据集上）

- **总体准确率**: 65-75%
- **训练速度**: 5-10分钟/epoch (4张A6000)
- **显存使用**: 20-30GB/GPU

### 各类别性能

通常情况下：
- **happy（开心）**: 准确率最高（80-85%）
- **neutral（中性）**: 准确率中等（70-75%）
- **disgust（厌恶）**: 准确率较低（50-60%，样本较少）

## 常见问题

### 1. CUDA out of memory

**解决方案**: 减小批次大小

```bash
python emotion_recognition_multi_gpu.py --batch_size 64
```

### 2. 训练速度慢

**解决方案**:
- 确保使用所有GPU
- 增加 `num_workers` 参数
- 检查数据加载是否为瓶颈

### 3. 模型准确率低

**解决方案**:
- 增加训练轮数
- 调整学习率
- 尝试不同的数据增强策略
- 使用更大的模型（如ResNet101）

### 4. NCCL错误

**解决方案**:
- 检查NCCL环境变量
- 确保所有GPU可见
- 尝试使用 `gloo` 后端

### 5. 摄像头无法打开

**解决方案**:
- 确保摄像头未被其他程序占用
- 检查摄像头驱动是否正常
- 尝试使用不同的摄像头后端

### 6. 画面里字符乱码

**解决方案**:
- 脚本已默认使用英文标签显示，避免中文和emoji乱码

## 模型部署

### 本地部署

使用 `emotion_camera.py` 进行本地摄像头实时检测：

```bash
python emotion_camera.py
```

### 服务器部署

使用 `emotion_server.py` 和 `emotion_client.py` 进行客户端-服务器架构部署：

1. 在服务器端启动服务：
   ```bash
   python emotion_server.py
   ```

2. 在客户端运行：
   ```bash
   python emotion_client.py --host 服务器IP
   ```

### 集成到其他项目

```python
import torch
from torchvision import transforms
from PIL import Image

# 加载模型
model = torch.load('emotion_model_best.pth')
model.eval()

# 预测图像
image = Image.open('test.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1)
```

## 引用

数据集: FER2013
- Goodfellow, I. J., et al. (2015). "Challenges in representation learning: A report on three machine learning contests."

## 许可证

本项目采用 MIT 许可证。