import os
import cv2
import time
import yaml
import shutil
import argparse
import numpy as np
from ultralytics import YOLO

####
# python train_iterative.py --n_epoch 10
####

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#os.environ["OMP_NUM_THREADS"]='8'
#os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'


BBOX_CONFIDENCE = 0.7
NMS_THRESHOLD = 0.3


def read_keypoints(file_path):
    keypoints = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            keypoints.append([float(val) for val in data])
    return np.array(keypoints)


def write_keypoints(file_path, keypoints):
    with open(file_path, 'w') as file:
        for keypoint in keypoints:
            line = ' '.join(str(val) for val in keypoint) + '\n'
            file.write(line)
232,

def calculate_iou(box1, box2):
    # Convert YOLO format to (x1, y1, x2, y2)
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    # Calculate intersection coordinates
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate area of both boxes
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou


def nms(true_keypoints, predict_keypoints, threshold):
    if len(predict_keypoints) == 0:
        return []
    
    picked = []
    for i in range(len(predict_keypoints)):

        iouList = []
        idx, rcx, rcy, rw, rh = predict_keypoints[i][:5]
        
        for gt in true_keypoints:
            gt_idx, gt_rcx, gt_rcy, gt_rw, gt_rh = gt[:5]
            iou = calculate_iou([rcx, rcy, rw, rh], [gt_rcx, gt_rcy, gt_rw, gt_rh])
            iouList.append(iou)
        
        #print(iouList)
        checkList = [1 if a>threshold else 0 for a in iouList]
        #print(np.sum(checkList))
        if np.sum(checkList) == 0:
            picked.append(predict_keypoints[i])
        else:
            pass
        
    return np.array(picked)


# Function to parse the data and extract bounding box coordinates
def parse_data(data):
    boxes = []
    keypoints = []
    for line in data:
        # Extract object class (assuming it's the first value)
        obj_class = int(line[0])
        # Extract bounding box coordinates
        box = [float(coord) for coord in line[1:5]]
        boxes.append((obj_class, box))
        # Extract keypoints
        keypoints.append([float(coord) for coord in line[5:]])

    return boxes, keypoints


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    # first
    parser.add_argument('--model_name', type=str, choices=["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", 
                                                           "yolov8l-pose.pt", "yolov8x-pose.pt", "yolov8x-pose-p6.pt"], 
                                                           default='yolov8n-pose.pt', help='initial model name')

    # setup (fix for all time)
    parser.add_argument('--yaml_path', type=str, default='./facial.yaml', help='The yaml path')
    parser.add_argument('--n_epoch', type=int, default=300, help='Total number of training epochs.')
    parser.add_argument('--n_patience', type=int, default=100, help='Number of epochs to wait without improvement in validation metrics before early stopping the training.')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--n_worker', type=int, default=8, help='Number of workers')
    parser.add_argument('--save_path', type=str, default='./runs/facial/', help='Save path')

    # predict (will change)
    parser.add_argument('--curr_iter', type=int, default=0, help='current iteration')
    parser.add_argument('--predict_weight', type=str, default='', help='previous best weight')
    parser.add_argument('--training_folder', type=str, default='../../datasets/v0/', help='predict on training data')
    parser.add_argument('--generate_folder', type=str, default='../../datasets/', help='generate new training data structure')
    
    return parser.parse_known_args()[0] if known else parser.parse_args()

opt = parse_opt()

i = opt.curr_iter

if __name__ == '__main__':

    print('='*160)
    print('iter:', i)
    print('='*160)


    if i == 0:
        ################ Step 1: Load a pretrained model ################
        model = YOLO(opt.model_name)
        print('='*80)
        print('pretrained model:', opt.model_name)
        print('='*80)

        ################ Step 2: Transfer learning ################
        # output: ./runs/facial/train/
        model.train(data=opt.yaml_path, epochs=opt.n_epoch, patience=opt.n_patience, batch=opt.bs, imgsz=opt.imgsz, device=0, workers=opt.n_worker, project=opt.save_path)

        ################ Step 3: Load a trained model ################
        current_best =  './runs/facial/train/weights/best.pt'
        model = YOLO(current_best)
        print('='*80)
        print('trained model:', current_best)
        print('='*80)

        ################ Step 4: Predict on training data ################
        # arguments: https://docs.ultralytics.com/modes/predict/
        # output: train: './runs/facial/predict/', val: './runs/facial/predict2/'
        # error: https://github.com/ultralytics/ultralytics/issues/1713
        # solution: https://github.com/ultralytics/ultralytics/issues/2930
        ct = 0
        datasetPath = opt.training_folder    # e.g. '../../datasets/v0/'
        imagesPath = datasetPath + 'images/'    # e.g. '../../datasets/v0/images/'
        imagesList = os.listdir(imagesPath)
        imagesList = sorted(imagesList)    # ['train', 'val']
        for foldern in imagesList:
            dataPath = imagesPath + foldern + '/'    # e.g. '../../datasets/v0/images/train/'
            dataList = os.listdir(dataPath)
            dataList = sorted(dataList)
            for r in model(dataPath, save_txt=True, stream=True, project=opt.save_path, name='predict', conf=BBOX_CONFIDENCE):
                pass

            ################ Step 5: Implement NMS on new and old labels ################
            # output: '../../datasets/v1/'
            trueLabelPath = datasetPath + 'labels/' + foldern + '/'    # e.g. '../../datasets/v0/labels/train/'
            if ct == 0:
                predictLabelPath = opt.save_path + 'predict/' + 'labels/'    # e.g. './runs/facial/predict/labels/'
            else:
                predictLabelPath = opt.save_path + 'predict' + str((i*2+1)+ct) + '/labels/'    # e.g. './runs/facial/predict2/labels/'

            savePath = opt.generate_folder + 'v' + str(i+1) + '/'    # e.g. '../../datasets/v1/'
            saveImagePath = savePath + 'images/'    # e.g. '../../datasets/v1/images/'
            saveImageDataPath = saveImagePath + foldern + '/'    # e.g. '../../datasets/v1/images/train/'
            saveLabelPath = savePath + 'labels/'    # e.g. '../../datasets/v1/labels/'
            saveLabelDataPath = saveLabelPath + foldern + '/'    # e.g. '../../datasets/v1/labels/train/'
            os.makedirs(savePath, exist_ok=True)
            os.makedirs(saveImagePath, exist_ok=True)
            os.makedirs(saveImageDataPath, exist_ok=True)
            os.makedirs(saveLabelPath, exist_ok=True)
            os.makedirs(saveLabelDataPath, exist_ok=True)

            for fn in dataList:
                ln = fn[:-3] + 'txt'
                tlp = trueLabelPath + ln    # true label path

                try:
                    plp = predictLabelPath + ln    # predict label path

                    # Read keypoints from files
                    keypoints1 = read_keypoints(tlp)
                    keypoints2 = read_keypoints(plp)

                    # other than original keypoints
                    new_keypoints = nms(keypoints1, keypoints2, NMS_THRESHOLD)

                    # original keypoints + new keypoints
                    if len(new_keypoints) == 0:
                        all_keypoints = keypoints1
                    else:
                        all_keypoints = np.vstack((keypoints1, new_keypoints))

                except:
                    # Read keypoints from files
                    keypoints1 = read_keypoints(tlp)
                    all_keypoints = keypoints1

                # Write filtered keypoints to a new file
                write_keypoints(saveLabelDataPath + ln, all_keypoints)
                print("Non-Maximum Suppression completed. Filtered keypoints saved to:", saveLabelDataPath + ln)

                # copy image into new dataset folder
                fp = dataPath + fn
                shutil.copy(fp, saveImageDataPath + fn)

            ct += 1

    else:
        ################ Step 1: write .yaml file ################
        # Define YAML content
        yaml_content = f"path: '/home/wish/pro/ICME2024/datasets/v{i}/'\n\
train: 'images/train/'\n\
val: 'images/val/'\n\
#test: 'images/test/'\n\
\n\
# Keypoints\n\
kpt_shape: [51, 3] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)\n\
flip_idx: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0,     # eyebrow\n\
           10, 11, 12, 13,     # vertical line for nose\n\
           18, 17, 16, 15, 14,     # horizontal line for nose\n\
           28, 27, 26, 25, 30, 29,     # left eye\n\
           22, 21, 20, 19, 24, 23,     # right eye\n\
           37, 36, 35, 34, 33, 32, 31,     # upper lip\n\
           42, 41, 40, 39, 38,     # lower lip\n\
           47, 46, 45, 44, 43,     # upper tooth\n\
           50, 49, 48]    # lower tooth\n\
\n\
# Classes dictionary\n\
names:\n\
  0: face\n"

        # File path to write the YAML file
        file_path = './facial_v'+ str(i) +'.yaml'

        # Writing data to YAML file
        with open(file_path, 'w') as file:
            file.write(yaml_content)

        print(f"YAML content has been written to {file_path}")

        ################ Step 1: Load a pretrained model ################
        if i == 1:
            previous_best =  './runs/facial/train/weights/best.pt'
        else:
            previous_best =  './runs/facial/train' + str(i) + '/weights/best.pt'
        model = YOLO(previous_best)
        print('='*80)
        print('pretrained model:', previous_best)
        print('='*80)

        ################ Step 2: Finetune on previous best model ################
        # output: ./runs/facial/train2/
        model.train(data=file_path, epochs=opt.n_epoch, patience=opt.n_patience, batch=opt.bs, imgsz=opt.imgsz, device=0, workers=opt.n_worker, project=opt.save_path)

        ################ Step 3: Load a trained model ################
        current_best =  './runs/facial/train' + str(i+1) + '/weights/best.pt'
        model = YOLO(current_best)
        print('='*80)
        print('trained model:', current_best)
        print('='*80)

        ################ Step 4: Predict on training data ################
        # arguments: https://docs.ultralytics.com/modes/predict/
        # output: train: './runs/facial/predict3/', val: './runs/facial/predict4/'
        ct = 0
        datasetPath = '../../datasets/v' + str(i) + '/'    # e.g. '../../datasets/v1/'
        imagesPath = datasetPath + 'images/'    # e.g. '../../datasets/v1/images/'
        imagesList = os.listdir(imagesPath)
        imagesList = sorted(imagesList)    # ['train', 'val']
        for foldern in imagesList:
            dataPath = imagesPath + foldern + '/'    # e.g. '../../datasets/v1/images/train/'
            dataList = os.listdir(dataPath)
            dataList = sorted(dataList)
            for r in model(dataPath, save_txt=True, stream=True, project=opt.save_path, name='predict', conf=BBOX_CONFIDENCE):
                pass

            ################ Step 5: Implement NMS on new and old labels ################
            # output: '../../datasets/v1/'
            trueLabelPath = datasetPath + 'labels/' + foldern + '/'    # e.g. '../../datasets/v1/labels/train/'
            trueLabelList = os.listdir(trueLabelPath)
            trueLabelList = sorted(trueLabelList)

            predictLabelPath = opt.save_path + 'predict' + str((i*2+1)+ct) + '/labels/'    # e.g. './runs/facial/predict3/labels/'

            savePath = opt.generate_folder + 'v' + str(i+1) + '/'    # e.g. '../../datasets/v2/'
            saveImagePath = savePath + 'images/'    # e.g. '../../datasets/v2/images/'
            saveImageDataPath = saveImagePath + foldern + '/'    # e.g. '../../datasets/v2/images/train/'
            saveLabelPath = savePath + 'labels/'    # e.g. '../../datasets/v2/labels/'
            saveLabelDataPath = saveLabelPath + foldern + '/'    # e.g. '../../datasets/v2/labels/train/'
            os.makedirs(savePath, exist_ok=True)
            os.makedirs(saveImagePath, exist_ok=True)
            os.makedirs(saveImageDataPath, exist_ok=True)
            os.makedirs(saveLabelPath, exist_ok=True)
            os.makedirs(saveLabelDataPath, exist_ok=True)

            for fn in dataList:
                ln = fn[:-3] + 'txt'
                tlp = trueLabelPath + ln    # true label path

                try:
                    plp = predictLabelPath + ln    # predict label path

                    # Read keypoints from files
                    keypoints1 = read_keypoints(tlp)
                    keypoints2 = read_keypoints(plp)

                    # other than original keypoints
                    new_keypoints = nms(keypoints1, keypoints2, NMS_THRESHOLD)

                    # original keypoints + new keypoints
                    if len(new_keypoints) == 0:
                        all_keypoints = keypoints1
                    else:
                        all_keypoints = np.vstack((keypoints1, new_keypoints))

                except:
                    # Read keypoints from files
                    keypoints1 = read_keypoints(tlp)
                    all_keypoints = keypoints1

                # Write filtered keypoints to a new file
                write_keypoints(saveLabelDataPath + ln, all_keypoints)
                print("Non-Maximum Suppression completed. Filtered keypoints saved to:", saveLabelDataPath + ln)

                # copy image into new dataset folder
                fp = dataPath + fn
                shutil.copy(fp, saveImageDataPath + fn)

            ct += 1
