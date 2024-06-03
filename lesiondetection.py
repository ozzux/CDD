from ultralytics import YOLO

model = YOLO('best.pt')

model.predict(source="IMG SOURCE",show=True, conf=0.4, save=True) # substitute the image location

input("Press enter to exit.")