from ultralytics import YOLO

model = YOLO('best.pt')

model.predict(source="C:/Users/osama/Documents/github/DICA/valid/images/6e8ce1116_jpg.rf.4ed74eb49bfe43b2c846e0964614c509.jpg", save=True, show=True)