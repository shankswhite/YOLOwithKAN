import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    # model = YOLO('/root/runs/detect/train6/weights/last.pt')
    model = YOLO('/mnt/d/project/ultralytics-8.3.3/ultralytics/cfg/models/11/yolo11.yaml')
    # model = YOLO('/mnt/d/project/ultralytics-8.3.3/runs/train/train/weights/last.pt       ')

    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/mnt/d/project/ultralytics-8.3.3/ultralytics/cfg/datasets/coco.yaml',
                cache=False,
                imgsz=640,
                epochs=600,
                batch=64,
                # close_mosaic=0,
                workers=5,
                device='0,1',
                pretrained=False,
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True,
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                plots=True,
                amp=False,
                )