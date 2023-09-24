import cv2
import time
import imutils
import numpy as np

from imutils.video import FPS
from imutils.video import VideoStream

#Initialize Objects and corresponding colors which the model can detect
labels = ["background", "aeroplane", "bicycle", "bird", 
"boat","bottle", "bus", "car", "cat", "chair", "cow", 
"diningtable","dog", "horse", "motorbike", "person", "pottedplant", 
"sheep","sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

#Loading Caffe Model
print('[Status] Loading Model...')
# Load the model and the prototxt file
model = "Object_Detection_Caffe\Caffe\SSD_MobileNet.caffemodel"
proto = "Object_Detection_Caffe\Caffe\SSD_MobileNet_prototxt.txt"
nn= cv2.dnn.readNetFromCaffe(proto, model)


while True:

    input_type = input("Choose input type:\n1.image\n2.video\n3.live\n:")

    if input_type =="1" :
        # Read the input image
        image_path = input("Enter path to image: ")
        frame = cv2.imread(image_path)

        # Resize Frame to 400 pixels
        frame = imutils.resize(frame, width=400)

        # Converting Frame to Blob
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # Passing Blob through network to detect and predict
        nn.setInput(blob)
        detections = nn.forward()

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):

            # Extracting the confidence of predictions
            confidence = detections[0, 0, i, 2]

            # Filtering out weak predictions
            if confidence > 0.7:

                # Extracting the index of the labels from the detection
                # Computing the (x,y) - coordinates of the bounding box        
                idx = int(detections[0, 0, i, 1])

                # Extracting bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Drawing the prediction and bounding box
                label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

        cv2.imshow("Frame", frame)
        cv2.waitKey(5000)

    elif input_type == "2":
        
        video_path = input("Enter path to video: ")
        vs = cv2.VideoCapture(video_path)

        fps = FPS().start()

        while True:
            ret, frame = vs.read()

            if not ret:
                break

            # Resize Frame to 400 pixels
            frame = imutils.resize(frame, width=400)
            (h, w) = frame.shape[:2]

            # Converting Frame to Blob
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

            # Passing Blob through network to detect and predict
            nn.setInput(blob)
            detections = nn.forward()

            # Loop over the detections
            for i in np.arange(0, detections.shape[2]):

                # Extracting the confidence of predictions
                confidence = detections[0, 0, i, 2]

                # Filtering out weak predictions
                if confidence > 0.7:

                    # Extracting the index of the labels from the detection
                    # Computing the (x,y) - coordinates of the bounding box        
                    idx = int(detections[0, 0, i, 1])

                    # Extracting bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Drawing the prediction and bounding box
                    label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)

                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            fps.update()

        fps.stop()

        print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
        print("[Info] Approximate FPS:  {:.2f}".format(fps.fps()))

        cv2.destroyAllWindows()
        vs.release()

    elif input_type == "3":
       
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        fps = FPS().start()

        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            (h, w) = frame.shape[:2]

            # Converting Frame to Blob
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

            # Passing Blob through network to detect and predict
            nn.setInput(blob)
            detections = nn.forward()

            # Loop over the detections
            for i in np.arange(0, detections.shape[2]):

                # Extracting the confidence of predictions
                confidence = detections[0, 0, i, 2]

                # Filtering out weak predictions
                if confidence > 0.7:
   
                    idx = int(detections[0, 0, i, 1])

                    # Extracting bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Drawing the prediction and bounding box
                    label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)

                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            fps.update()

        fps.stop()

        print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
        print("[Info] Approximate FPS:  {:.2f}".format(fps.fps()))

        cv2.destroyAllWindows()
        vs.stop()

    else:
        print("Invalid input type")
