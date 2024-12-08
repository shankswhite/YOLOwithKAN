import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

# 精度小数点保留位数修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第五点

if __name__ == '__main__':
    model = YOLO('11-official.pt')
    model.val(data='/mnt/d/project/ultralytics-8.3.3/ultralytics/cfg/datasets/coco.yaml',
              split='val',
              imgsz=640,
              batch=64,
              # iou=0.7,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )

    # model = YOLO("11-official.pt")
    # metrics = model.val()  # no arguments needed, dataset and settings remembered
    # metrics.box.map  # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps