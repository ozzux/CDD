from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os



for imag in os.listdir('convdata/ADEM'):
    print(imag)
    image_path = 'convdata/ADEM/'+imag

    img = cv2.imread(image_path)
    H, W, _ = img.shape

    model = YOLO('segmentationmodel.pt')
    results = model.predict(img)
    try:
        for result in results:
            # get array results
            masks = result.masks.data
            boxes = result.boxes.data
            # extract classes
            clss = boxes[:, 5]
            # get indices of results where class is 0 (people in COCO)
            people_indices = torch.where(clss == 0)
            # use these indices to extract the relevant masks
            people_masks = masks[people_indices]
            # scale for visualizing results
            people_mask = torch.any(people_masks, dim=0).int() * 255
            # save to file
            cv2.imwrite('predicted_masks/ADEM/'+imag, people_mask.cpu().numpy())
    except:
        print('No Lesions Detected')
        pass

for imag in os.listdir('convdata/MS'):
    print(imag)
    image_path = 'convdata/MS/'+imag

    img = cv2.imread(image_path)
    H, W, _ = img.shape

    model = YOLO('segmentationmodel.pt')
    results = model.predict(img)
    try:
        for result in results:
            # get array results
            masks = result.masks.data
            boxes = result.boxes.data
            # extract classes
            clss = boxes[:, 5]
            # get indices of results where class is 0 (people in COCO)
            people_indices = torch.where(clss == 0)
            # use these indices to extract the relevant masks
            people_masks = masks[people_indices]
            # scale for visualizing results
            people_mask = torch.any(people_masks, dim=0).int() * 255
            # save to file
            cv2.imwrite('predicted_masks/MS/'+imag, people_mask.cpu().numpy())
    except:
        print('No Lesions Detected')
        pass