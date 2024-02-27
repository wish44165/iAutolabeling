import argparse
from ultralytics import YOLO

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='./runs/facial/train/weights/best.pt', help='trained model')
    parser.add_argument('--save_path', type=str, default='./runs/facial', help='Save path')
    return parser.parse_known_args()[0] if known else parser.parse_args()

opt = parse_opt()

# Load a model
model = YOLO(opt.weight)  # load a custom model

# Validate the model
metrics = model.val(project=opt.save_path, save=True, save_txt=True)  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

# https://stackoverflow.com/questions/75239330/yolov8-how-i-can-to-predict-and-save-the-image-with-boxes-on-the-objects-with-p