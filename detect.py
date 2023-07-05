import cv2
import collections
import numpy as np
import math
import time
#import pafy
#import yt_dlp as youtube_dl
import tracker.py

#cv2.setNumThreads(4)

cap = cv2.VideoCapture("video.mp4")
input_size = 320

confThreshold = 0.2
nmsThreshold = 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2


classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
# print(classNames)
# print(len(classNames))

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


#previous_frame = None
#previous_keypoints = None

prev_frame = None
prev_gray = None
prev_centers = {}
prev_time = None


class detect:
    def __init__(self):
        self.boxes = []
        self.classIds = []
        self.confidence_scores = []
        self.detection = []
        
    def postProcess(self, outputs, img):
        global detected_classNames
        global prev_frame, prev_gray, prev_centers, prev_time
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        track = tracker()
        for output in outputs:
            for det in output:
                scores = det[5:]
                self.classId = np.argmax(scores)
                confidence = scores[self.classId]
                if self.classId in required_class_index:
                    if confidence > confThreshold:
                        # print(self.classId)
                        w, h = int(det[2] * width), int(det[3] * height)
                        x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                        self.boxes.append([x, y, w, h])
                        self.classIds.append(self.classId)
                        self.confidence_scores.append(float(confidence))

        # Apply Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(self.boxes, self.confidence_scores, confThreshold, nmsThreshold)
        # print(self.classIds)
        for i in indices.flatten():
            x, y, w, h = self.boxes[i][0], self.boxes[i][1], self.boxes[i][2], self.boxes[i][3]
            # print(x,y,w,h)

            color = [int(c) for c in colors[self.classIds[i]]]
            name = classNames[self.classIds[i]]
            detected_classNames.append(name)
            cv2.putText(img, f'{name.upper()} {int(self.confidence_scores[i] * 100)}%',(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            self.detection.append([x, y, w, h, required_class_index.index(self.classIds[i])])

        if prev_frame is not None:
            # Calculate the optical flow for each object
            for i in indices.flatten():
                x, y, w, h = self.boxes[i][0], self.boxes[i][1], self.boxes[i][2], self.boxes[i][3]
                center = find_center(x, y, w, h)
                cx, cy = center

                if id in prev_centers:
                    prev_cx, prev_cy = prev_centers[id]

                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flow_x = flow[int(prev_cy), int(prev_cx), 0]
                    flow_y = flow[int(prev_cy), int(prev_cx), 1]
                    magnitude = math.sqrt(flow_x ** 2 + flow_y ** 2)
                    elapsed_time = time.time() - prev_time
                    
                    """
                    pixels_per_meter = 1.0  # Modify this value according to the real-world scale
                    meters_per_sec = magnitude * pixels_per_meter / elapsed_time
                    km_per_hr = meters_per_sec * 3.6
                    
                    cv2.putText(img, f"{km_per_hr:.2f} km/h", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    """
                    #uncomment the above LOC to estim
                    speed = magnitude / elapsed_time
                    cv2.putText(img, f"{speed:.2f} px/s", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    
                prev_centers[id] = center

        prev_frame = img.copy()
        prev_gray = gray.copy()
        prev_time = time.time()
        boxes_ids = track.update(self.detection)

    def realTime(self):
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame,(0,0),None,0.5,0.5)
            ih, iw, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

            net.setInput(blob)
            layersNames = net.getLayerNames()
            outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
            outputs = net.forward(outputNames)

            process = detect()
            process.postProcess(outputs, frame)

            cv2.imshow('Output', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
