import cv2
import numpy as np
import math
import time
from track import tracker

def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

class detect:
    def __init__(self, net, classNames, colors, required_class_index, confThreshold, nmsThreshold, PIXELS_PER_METER, SPEED_MULTIPLIER):
        self.net = net
        self.classNames = classNames
        self.colors = colors
        self.required_class_index = required_class_index
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.PIXELS_PER_METER = PIXELS_PER_METER
        self.SPEED_MULTIPLIER = SPEED_MULTIPLIER
        
        self.boxes = []
        self.classIds = []
        self.confidence_scores = []
        self.detection = []
        self.track = tracker()
        self.prev_positions = {}
        self.prev_time = {}
        self.speed_history = {}
        self.position_history = {}
        self.frame_count = 0
        
    def calculate_speed_kmh(self, object_id, current_pos, current_time, frame_height):
        if object_id not in self.prev_positions or object_id not in self.prev_time:
            self.prev_positions[object_id] = current_pos
            self.prev_time[object_id] = current_time
            if object_id not in self.position_history:
                self.position_history[object_id] = []
            return 0
        
        prev_pos = self.prev_positions[object_id]
        prev_time = self.prev_time[object_id]
        
        if object_id not in self.position_history:
            self.position_history[object_id] = []
        self.position_history[object_id].append(current_pos)
        
        if len(self.position_history[object_id]) > 10:
            self.position_history[object_id] = self.position_history[object_id][-10:]
        
        distance_pixels = math.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
        
        y_position = current_pos[1]
        perspective_factor = 1.0 + (y_position / frame_height) * 0.5
        distance_pixels *= perspective_factor
        
        time_diff = current_time - prev_time
        
        if time_diff <= 0 or distance_pixels < 2:
            return self.get_smoothed_speed(object_id, 0)
        
        distance_meters = distance_pixels / self.PIXELS_PER_METER
        speed_ms = distance_meters / time_diff
        speed_kmh = speed_ms * 3.6 * self.SPEED_MULTIPLIER
        
        speed_kmh = max(0, min(speed_kmh, 120))
        
        self.prev_positions[object_id] = current_pos
        self.prev_time[object_id] = current_time
        
        return self.get_smoothed_speed(object_id, speed_kmh)
    
    def get_smoothed_speed(self, object_id, new_speed):
        if object_id not in self.speed_history:
            self.speed_history[object_id] = []
        
        if new_speed > 0:
            self.speed_history[object_id].append(new_speed)
        
        if len(self.speed_history[object_id]) > 15:
            self.speed_history[object_id] = self.speed_history[object_id][-15:]
        
        if len(self.speed_history[object_id]) == 0:
            return 0
        
        speeds = self.speed_history[object_id]
        if len(speeds) >= 3:
            median_speed = sorted(speeds)[len(speeds)//2]
            filtered_speeds = [s for s in speeds if abs(s - median_speed) < median_speed * 0.4]
            if filtered_speeds:
                speeds = filtered_speeds
        
        weights = [i + 1 for i in range(len(speeds))]
        weighted_avg = sum(s * w for s, w in zip(speeds, weights)) / sum(weights)
        
        return weighted_avg
        
    def postProcess(self, outputs, img):
        self.boxes = []
        self.classIds = []
        self.confidence_scores = []
        self.detection = []
        
        height, width = img.shape[:2]
        current_time = time.time()
        self.frame_count += 1

        for output in outputs:
            for det in output:
                scores = det[5:]
                self.classId = np.argmax(scores)
                confidence = scores[self.classId]
                if self.classId in self.required_class_index:
                    if confidence > self.confThreshold:
                        w, h = int(det[2] * width), int(det[3] * height)
                        x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                        self.boxes.append([x, y, w, h])
                        self.classIds.append(self.classId)
                        self.confidence_scores.append(float(confidence))

        if len(self.boxes) > 0:
            indices = cv2.dnn.NMSBoxes(self.boxes, self.confidence_scores, self.confThreshold, self.nmsThreshold)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = self.boxes[i][0], self.boxes[i][1], self.boxes[i][2], self.boxes[i][3]
                    self.detection.append([x, y, w, h, self.required_class_index.index(self.classIds[i])])

        if len(self.detection) > 0:
            boxes_ids = self.track.update(self.detection)
            
            for box_id in boxes_ids:
                x, y, w, h, object_id, index = box_id
                
                cx, cy = find_center(x, y, w, h)
                
                speed_kmh = self.calculate_speed_kmh(object_id, (cx, cy), current_time, height)
                
                color = [int(c) for c in self.colors[self.classIds[index] if index < len(self.classIds) else 0]]
                name = self.classNames[self.classIds[index] if index < len(self.classIds) else 0]
                
                speed_color = (0, 0, 255) if speed_kmh > 80 else (0, 255, 0)
                
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                cv2.putText(img, f'{name.upper()} {int(self.confidence_scores[index] * 100)}%' if index < len(self.confidence_scores) else f'{name.upper()}',
                           (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                if speed_kmh > 10 and len(self.speed_history.get(object_id, [])) >= 3:
                    cv2.putText(img, f"Speed: {speed_kmh:.0f} km/h", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_color, 1)
                
                cv2.putText(img, f"ID: {object_id}", (x, y + h + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
