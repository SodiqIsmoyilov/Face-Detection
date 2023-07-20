from ultralytics import YOLO
import supervision as sv
import numpy as np

import torch
import cv2


def main():
    video = cv2.VideoCapture(0)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = YOLO("models/best.pt")
    while True:

        ret, frame = video.read()

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        results = model.predict(frame, conf=0.7, stream=True)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)
                print(r)
                cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)

        cv2.imshow("input", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()










