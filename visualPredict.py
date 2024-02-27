import os
import cv2
import numpy as np


# Function to parse the data and extract bounding box coordinates
def parse_data(data):
    boxes = []
    keypoints = []
    for line in data:
        line = line.strip().split()
        # Extract object class (assuming it's the first value)
        obj_class = int(line[0])
        # Extract bounding box coordinates
        box = [float(coord) for coord in line[1:5]]
        boxes.append((obj_class, box))
        # Extract keypoints
        keypoints.append([float(coord) for coord in line[5:]])

    return boxes, keypoints

folderPath = './runs/facial/predict/'
imagePath = folderPath
labelPath = folderPath + 'labels/'
imageList = os.listdir(imagePath)
imageList.remove('labels')
imageList = sorted(imageList)
labelList = os.listdir(labelPath)
labelList = sorted(labelList)
#print(imageList)
#print(labelList)



for fn in imageList:
    fp = imagePath + fn
    lp = labelPath + fn[:-3] + 'txt'

    with open(lp, 'r') as file:
        data = file.readlines()
        # Parse the data
        boxes, keypoints = parse_data(data)
        #print(boxes, keypoints)

        # Read the image
        image = cv2.imread(fp)

        # Get the height and width of the image
        height, width, _ = image.shape

        # Plot bounding boxes
        for obj_class, box in boxes:
            x, y, w, h = box
            # Convert (x, y, w, h) to pixel coordinates
            xmin = int((x - w / 2) * width)
            ymin = int((y - h / 2) * height)
            xmax = int((x + w / 2) * width)
            ymax = int((y + h / 2) * height)
            # Draw bounding box rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            cv2.putText(image, f'Class {obj_class}', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

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
        cv2.imshow('Image with Bounding Boxes and Keypoints', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()