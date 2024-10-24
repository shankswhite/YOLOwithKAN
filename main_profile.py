import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('ultralytics/cfg/models/v10/yolov10n.yaml')
    model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    model.fuse()