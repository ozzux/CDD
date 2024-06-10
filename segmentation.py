import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
import cv2
from segment_anything import *

imagePath = os.path.join("All Data","MS","MS_1.png")

model = YOLO('yolomodel.pt')

results = model.predict(source=imagePath, conf=0.3)
predicted_boxes = results[0].boxes.xyxy

image_bgr = cv2.imread(imagePath, cv2.IMREAD_COLOR)

for box in predicted_boxes:
    cv2.rectangle(image_bgr, (int(box[0]),int(box[1])),(int(box[2]),int(box[3])) , color=(0,255,0), thickness=2)
cv2.imshow("YOLO IMAGES",image_bgr)

cv2.waitKey()

sam = sam_model_registry["vit_h"](checkpoint="/sam_vit_h_4b8939.pth").to(device=torch.device('cuda:0'))

mask_predictor = SamPredictor(sam)

transformed_boxes = mask_predictor.transform.apply_boxes_torch(predicted_boxes, image_bgr.shape[:2])

mask_predictor.set_image(image_bgr)
masks, scores, logits = mask_predictor.predict_torch(
   boxes = transformed_boxes,
   multimask_output=False,
   point_coords=None,
   point_labels=None
)

final_mask = None
for i in range(len(masks) - 1):
  if final_mask is None:
    final_mask = np.bitwise_or(masks[i][0], masks[i+1][0])
  else:
    final_mask = np.bitwise_or(final_mask, masks[i+1][0])

plt.figure(figsize=(10, 10))
plt.imshow(image_bgr)
plt.imshow(final_mask, cmap='gray', alpha=0.7)
plt.show()