import socket
import pickle
import cv2
import numpy as np
import io
from PIL import Image


def capture_and_send(host='localhost', port=5000):
    """从摄像头捕获画面并发送到服务器"""
    # 连接到服务器
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
        print(f"已连接到服务器 {host}:{port}")
    except Exception as e:
        print(f"无法连接到服务器: {str(e)}")
        return
    
    # 打开摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("无法打开摄像头")
        client_socket.close()
        return
    
    print("按 'q' 键退出")
    
    while True:
        try:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                continue
            
            # 将帧转换为字节数据
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            image_data = buffer.tobytes()
            
            # 序列化并发送数据
            client_socket.send(pickle.dumps(image_data))
            
            # 接收服务器的响应
            response = client_socket.recv(1024)
            if not response:
                break
            
            # 反序列化响应
            result = pickle.loads(response)
            
            # 在帧上绘制结果
            cv2.putText(frame, f'{result["emoji"]} {result["emotion_cn"]}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'置信度: {result["confidence"]:.2f}', (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 显示帧
            cv2.imshow('表情识别', frame)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            continue
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    client_socket.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='表情识别客户端')
    parser.add_argument('--host', type=str, default='localhost',
                       help='服务器主机')
    parser.add_argument('--port', type=int, default=5000,
                       help='服务器端口')
    
    args = parser.parse_args()
    
    capture_and_send(args.host, args.port)