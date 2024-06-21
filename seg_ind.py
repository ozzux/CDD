from ultralytics import YOLO
import cv2
import torch

def seg(img_path):
    model = YOLO('segmentationmodel.pt')

    results = model.predict(source=img_path)

    result = results[0]

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
    return people_mask.cpu().numpy()

