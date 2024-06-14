from ultralytics import YOLO
import cv2

model = YOLO('segmentmodel.pt')

results = model.predict(source="All Data/MS")

i = 0

for result in results:
  i += 1
  masks = result[0].masks.data.numpy()

  mask = masks[0]

  for msk in masks:
    mask += msk

  mask = mask * 255

  results[0].show()
  cv2.imwrite(str(i)+".png",mask)