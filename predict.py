import os
import argparse
from ultralytics import YOLO

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='./runs/facial/5_best.pt', help='trained model')
    parser.add_argument('--save_path', type=str, default='./runs/facial', help='Save path')
    parser.add_argument('--predict_folder', type=str, default='../datasets/v0/images/', help='data images')
    parser.add_argument('--save_folder', type=str, default='./runs/facial/', help='predicted labels')
    return parser.parse_known_args()[0] if known else parser.parse_args()

opt = parse_opt()

# Load a model
model = YOLO(opt.weight)  # load a custom model

dataPath = opt.predict_folder
dataList = os.listdir(dataPath)
dataList = sorted(dataList)
for foldern in dataList:
    imagePath = dataPath + foldern + '/'
    #imageList = os.listdir(imagePath)
    #imageList = sorted(imageList)

    results = model(imagePath, save=True, save_txt=True, project=opt.save_folder)

"""
trainPath = dataPath + 'images/train/'
trainList = os.listdir(trainPath)
trainList = sorted(trainList)
valPath = dataPath + 'images/val/'
valList = os.listdir(valPath)
valList = sorted(valList)
"""


# Predict with the model
#results = model(opt.predict_folder, save=True, save_txt=True, project=opt.save_path)