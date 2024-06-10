from ultralytics import YOLO

model = YOLO('yolomodel.pt')

model.predict(source="All Data\MS\MS_17.png", conf=0.4, save=True, show_labels=False) # substitute the image location
