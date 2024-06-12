from ultralytics import YOLO
import cv2

model = YOLO('segmentmodel.pt')

results = model.predict(source="All Data/MS/MS_20.png")

masks = results[0].masks.data.numpy()

mask = masks[0]

for msk in masks:
  mask += msk

mask = mask * 255

cv2.imshow('result', mask)
cv2.waitKey()

results[0].show()
cv2.waitKey()