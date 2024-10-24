from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")  # 加载预训练模型
    # 使用绝对路径并添加 r 表示原始字符串
    model.train(data="\\mnt\\d\\project\\ultralytics-8.3.3\\ultralytics\\cfg\\datasets\\coco.yaml", epochs=100, workers=16, batch=16, device=[0,1], cache=True)

if __name__ == '__main__':
    main()
