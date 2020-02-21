import numpy as np
import argparse
import cv2

# Xu ly tham so dau vao
parser = argparse.ArgumentParser(description='Use MobileNet SSD on Pi for object detection')
parser.add_argument("--vid_file", help="Duong dan den file video")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt")
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel")
args = parser.parse_args()

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

def cal_position(detections,i,cols,rows):
    # Lay class_id
    class_id = int(detections[0, 0, i, 1])

    # Tinh toan vi tri cua doi tuong
    xLeftBottom = int(detections[0, 0, i, 3] * cols)
    yLeftBottom = int(detections[0, 0, i, 4] * rows)
    xRightTop = int(detections[0, 0, i, 5] * cols)
    yRightTop = int(detections[0, 0, i, 6] * rows)


    heightFactor = frame.shape[0] / 300.0
    widthFactor = frame.shape[1] / 300.0


    xLeftBottom = int(widthFactor * xLeftBottom)
    yLeftBottom = int(heightFactor * yLeftBottom)
    xRightTop = int(widthFactor * xRightTop)
    yRightTop = int(heightFactor * yRightTop)

    return class_id, xLeftBottom, yLeftBottom, xRightTop, yRightTop

def do_detect(frame, net, classNames):

    # Resize anh ve 300x300
    frame_resized = cv2.resize(frame, (300, 300))

    # Doc blob va dua vao mang predict
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()

    # Xu ly output cua mang
    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]

    # Duyet qua cac object detect duoc
    for i in range(detections.shape[2]):
        # Lay gia tri confidence
        confidence = detections[0, 0, i, 2]
        # Neu vuot qua 0.5 threshold
        if confidence > 0.5:

            # Tinh toan vi tri cua doi tuong
            class_id, xLeftBottom, yLeftBottom, xRightTop, yRightTop = cal_position(detections,i, cols,rows)

            # Ve khung hinh chu nhat
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),  (0, 255, 0))

            # Ve label cua doi tuong
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return  frame

# Mo video hoac webcam
if args.vid_file:
    cap = cv2.VideoCapture(args.vid_file)
else:
    cap = cv2.VideoCapture(0)

# Load model
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

# Bat dau doc tu video/webcam
# Bien dem frame
i_frame = 0
while True:
    # Doc tung frame
    ret, frame = cap.read()
    # Tang bien dem
    i_frame +=1
    # Xu ly detection moi 20 frame de giam tai cho Pi
    if  i_frame%20==0:
        # Thuc hien detect
        frame = do_detect(frame,net,classNames)
        # Hien thi frame len man hinh
        cv2.imshow("frame", frame)

    # Neu nhan Esc thi thoat
    if cv2.waitKey(1) >= 0:
        break
