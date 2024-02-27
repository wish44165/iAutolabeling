import os
import cv2
import numpy as np

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

    # Sort keypoints by confidence score (assuming the last value is the confidence score)
    #keypoints = predict_keypoints[predict_keypoints[:, -1].argsort()[::-1]]

    #print(predict_keypoints)
    #print(predict_keypoints.shape)

    
    picked = []
    for i in range(len(predict_keypoints)):

        iouList = []
        idx, rcx, rcy, rw, rh = predict_keypoints[i][:5]
        #print(idx, rcx, rcy, rw, rh)
        
        for gt in true_keypoints:
            gt_idx, gt_rcx, gt_rcy, gt_rw, gt_rh = gt[:5]
            #print(gt_idx, gt_rcx, gt_rcy, gt_rw, gt_rh)
            #print(calculate_iou([rcx, rcy, rw, rh], [gt_rcx, gt_rcy, gt_rw, gt_rh]))
            iou = calculate_iou([rcx, rcy, rw, rh], [gt_rcx, gt_rcy, gt_rw, gt_rh])
            iouList.append(iou)
        
        #print(iouList)
        checkList = [1 if a>threshold else 0 for a in iouList]
        print(np.sum(checkList))
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


# File paths
trueLabelPath = '/home/wish/UoM/0thers/IEEE_ICME_2024_Grand_Challenges/src/datasets/v0/labels/train/'
imagePath = '/home/wish/UoM/0thers/IEEE_ICME_2024_Grand_Challenges/src/datasets/v0/images/train/'
predictLabelPath = './runs/facial/predict/labels/'

savePath = '/home/wish/UoM/0thers/IEEE_ICME_2024_Grand_Challenges/src/datasets/v1/'
os.makedirs(savePath, exist_ok=True)
saveTrainPath = savePath + 'train/'
os.makedirs(saveTrainPath, exist_ok=True)

trueLabelList = os.listdir(trueLabelPath)
trueLabelList = sorted(trueLabelList)

for ln in trueLabelList:
    tlp = trueLabelPath + ln    # true label path
    plp = predictLabelPath + ln    # predict label path
    fp = imagePath + ln[:-3] + 'png'
    

    # Read keypoints from files
    keypoints1 = read_keypoints(tlp)
    keypoints2 = read_keypoints(plp)

    print(keypoints1.shape)    # 0, rcx, rcy, rw, rh, rx, ry, vis
    print(keypoints2.shape)    # 0, rcx, rcy, rw, rh, rx, ry, conf

    # other than original keypoints
    new_keypoints = nms(keypoints1, keypoints2, NMS_THRESHOLD)
    
    # original keypoints + new keypoints
    if len(new_keypoints) == 0:
        all_keypoints = keypoints1
    else:
        all_keypoints = np.vstack((keypoints1, new_keypoints))



    ######## visualization ########

    # Read the image
    image = cv2.imread(fp)
    # Get the height and width of the image
    height, width, _ = image.shape



    boxes, keypoints = parse_data(keypoints1)

    # Plot bounding boxes
    for obj_class, box in boxes:
        x, y, w, h = box
        # Convert (x, y, w, h) to pixel coordinates
        xmin = int((x - w / 2) * width)
        ymin = int((y - h / 2) * height)
        xmax = int((x + w / 2) * width)
        ymax = int((y + h / 2) * height)
        # Draw bounding box rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 10)
        cv2.putText(image, f'Class {obj_class}', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 10, cv2.LINE_AA)

    # Plot keypoints
    for kpts in keypoints:
        for i in range(0, len(kpts), 3):
            x, y, visibility = kpts[i], kpts[i+1], kpts[i+2]
            if visibility > 0:  # Only plot visible keypoints
                # Convert keypoints to pixel coordinates
                kpt_x = int(x * width)
                kpt_y = int(y * height)
                # Draw keypoints
                cv2.circle(image, (kpt_x, kpt_y), 3, (0, 0, 255), -1)




    ## predict
    # Parse the data
    boxes, keypoints = parse_data(keypoints2)

    # Plot bounding boxes
    for obj_class, box in boxes:
        x, y, w, h = box
        # Convert (x, y, w, h) to pixel coordinates
        xmin = int((x - w / 2) * width)
        ymin = int((y - h / 2) * height)
        xmax = int((x + w / 2) * width)
        ymax = int((y + h / 2) * height)
        # Draw bounding box rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 155, 0), 6)
        cv2.putText(image, f'Class {obj_class}', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 155, 0), 6, cv2.LINE_AA)

    # Plot keypoints
    for kpts in keypoints:
        for i in range(0, len(kpts), 3):
            x, y, visibility = kpts[i], kpts[i+1], kpts[i+2]
            if visibility > 0:  # Only plot visible keypoints
                # Convert keypoints to pixel coordinates
                kpt_x = int(x * width)
                kpt_y = int(y * height)
                # Draw keypoints
                cv2.circle(image, (kpt_x, kpt_y), 3, (0, 155, 0), -1)




    ## new keypoints
    # Parse the data
    boxes, keypoints = parse_data(new_keypoints)

    # Plot bounding boxes
    for obj_class, box in boxes:
        x, y, w, h = box
        # Convert (x, y, w, h) to pixel coordinates
        xmin = int((x - w / 2) * width)
        ymin = int((y - h / 2) * height)
        xmax = int((x + w / 2) * width)
        ymax = int((y + h / 2) * height)
        # Draw bounding box rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(image, f'Class {obj_class}', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    # Plot keypoints
    for kpts in keypoints:
        for i in range(0, len(kpts), 3):
            x, y, visibility = kpts[i], kpts[i+1], kpts[i+2]
            if visibility > 0:  # Only plot visible keypoints
                # Convert keypoints to pixel coordinates
                kpt_x = int(x * width)
                kpt_y = int(y * height)
                # Draw keypoints
                cv2.circle(image, (kpt_x, kpt_y), 3, (255, 0, 0), -1)

    # Display the image
    cv2.imshow('Image with Bounding Boxes and Keypoints', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




    ################ after nms ################


    # Read the image
    image = cv2.imread(fp)
    # Get the height and width of the image
    height, width, _ = image.shape

    boxes, keypoints = parse_data(all_keypoints)

    # Plot bounding boxes
    for obj_class, box in boxes:
        x, y, w, h = box
        # Convert (x, y, w, h) to pixel coordinates
        xmin = int((x - w / 2) * width)
        ymin = int((y - h / 2) * height)
        xmax = int((x + w / 2) * width)
        ymax = int((y + h / 2) * height)
        # Draw bounding box rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
        cv2.putText(image, f'Class {obj_class}', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 4, cv2.LINE_AA)

    # Plot keypoints
    for kpts in keypoints:
        for i in range(0, len(kpts), 3):
            x, y, visibility = kpts[i], kpts[i+1], kpts[i+2]
            if visibility > 0:  # Only plot visible keypoints
                # Convert keypoints to pixel coordinates
                kpt_x = int(x * width)
                kpt_y = int(y * height)
                # Draw keypoints
                cv2.circle(image, (kpt_x, kpt_y), 3, (0, 255, 0), -1)

    # Display the image
    cv2.imshow('Image with New Bounding Boxes and Keypoints', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




    # Write filtered keypoints to a new file
    write_keypoints(saveTrainPath + ln, all_keypoints)

    print("Non-Maximum Suppression completed. Filtered keypoints saved to:", saveTrainPath + ln)