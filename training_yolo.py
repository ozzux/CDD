from ultralytics import YOLO

model = YOLO('best.pt')

results = model.predict(source="https://www.shutterstock.com/image-photo/aerial-view-cargo-ship-business-260nw-1677971977.jpg",show=True, save=True)