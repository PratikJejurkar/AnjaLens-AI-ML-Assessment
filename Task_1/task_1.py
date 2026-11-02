import json
import time
import math
import cv2
import torch
import numpy as np
import requests
import os
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

# --- Step 1: Download Sample Video & Create Config ---
video_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/worker-zone-detection.mp4"
video_path = "video/worker-zone-detection.mp4"

if not os.path.exists(video_path):
    print("Downloading sample factory video...")
    response = requests.get(video_url)
    with open(video_path, 'wb') as f:
        f.write(response.content)

# We use normalized coordinates (0.0 to 1.0) for the polygon so it scales to any video resolution
config_data = {
    "zone_polygon_normalized": [[0.3, 0.4], [0.9, 0.4], [0.9, 0.9], [0.3, 0.9]],
    "idle_rules": {
        "pixel_distance": 30,
        "time_seconds": 3.0
    }
}
with open('config.json', 'w') as f:
    json.dump(config_data, f, indent=4)

# --- Step 2: Load Config & Initialize Setup ---
with open('config.json', 'r') as f:
    config = json.load(f)

idle_dist_thresh = config["idle_rules"]["pixel_distance"]
idle_time_thresh = config["idle_rules"]["time_seconds"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hardware_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
print(f"Initializing YOLO11n on: {hardware_name}")

model = YOLO('yolo11n.pt') # Downloads the nano model automatically

# --- Step 3: Video Setup ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Convert normalized polygon coordinates to absolute pixel coordinates
abs_polygon = [[int(x * width), int(y * height)] for x, y in config["zone_polygon_normalized"]]
zone_poly = Polygon(abs_polygon)
np_polygon = np.array(abs_polygon, np.int32)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# --- Step 4: State Management ---
events = []
in_zone_ids = set()
position_history = {} # track_id: [(x, y, timestamp)]
idle_alerted = set()

start_time = time.time()
frame_count = 0

print("Processing video... This may take a moment.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    current_timestamp = frame_count / fps
    banner_text = "STATUS: NORMAL"
    banner_color = (0, 255, 0)

    # Draw the Restricted Zone Polygon
    cv2.polylines(frame, [np_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

    # Track using BoT-SORT for stable IDs
    results = model.track(frame, classes=[0], persist=True, tracker="botsort.yaml", verbose=False)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            # Calculate foot point (bottom-center of the bounding box)
            foot_x = (x1 + x2) / 2
            foot_y = y2
            foot_point = Point(foot_x, foot_y)
            
            # Draw bbox and foot point
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.circle(frame, (int(foot_x), int(foot_y)), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # --- Rule 1: Zone Entry Monitoring ---
            if zone_poly.contains(foot_point):
                if track_id not in in_zone_ids:
                    in_zone_ids.add(track_id)
                    events.append({
                        "event_type": "ZONE_ENTRY",
                        "track_id": int(track_id),
                        "timestamp": round(current_timestamp, 2),
                        "frame_number": frame_count,
                        "box": [round(float(x), 2) for x in box]
                    })
                banner_text = f"ZONE ENTRY ALERT: ID {track_id}"
                banner_color = (0, 0, 255)
            else:
                if track_id in in_zone_ids:
                    in_zone_ids.remove(track_id)

            # --- Rule 2: Idle Monitoring ---
            if track_id not in position_history:
                position_history[track_id] = []
            
            history = position_history[track_id]
            history.append((foot_x, foot_y, current_timestamp))
            
            # Prune history older than the threshold window
            history = [p for p in history if current_timestamp - p[2] <= idle_time_thresh]
            position_history[track_id] = history
            
            if len(history) > 1 and (current_timestamp - history[0][2]) >= (idle_time_thresh - 0.5):
                # Calculate maximum pixel distance moved within the time window
                max_dist = 0
                for i in range(len(history)):
                    for j in range(i+1, len(history)):
                        dist = math.hypot(history[i][0] - history[j][0], history[i][1] - history[j][1])
                        max_dist = max(max_dist, dist)
                
                if max_dist < idle_dist_thresh:
                    if track_id not in idle_alerted:
                        idle_alerted.add(track_id)
                        events.append({
                            "event_type": "IDLE_ALERT",
                            "track_id": int(track_id),
                            "timestamp": round(current_timestamp, 2),
                            "frame_number": frame_count,
                            "box": [round(float(x), 2) for x in box]
                        })
                    # Idle alert takes UI precedence if standing still inside a zone
                    banner_text = f"IDLE ALERT: ID {track_id}"
                    banner_color = (0, 165, 255)
                else:
                    if track_id in idle_alerted:
                        idle_alerted.remove(track_id)

    # Draw live event banner
    cv2.putText(frame, banner_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, banner_color, 3)
    out.write(frame)

cap.release()
out.release()

# --- Step 5: Hardware and Performance Logging ---
end_time = time.time()
avg_fps = frame_count / (end_time - start_time)

print("-" * 30)
print("PROCESSING COMPLETE")
print(f"Hardware Used: {hardware_name}")
print(f"Average Processing FPS: {avg_fps:.2f}")
print("-" * 30)

# Write events to JSON
with open('events.json', 'w') as f:
    json.dump(events, f, indent=4)
print(f"Logged {len(events)} events to 'events.json'")
print("Annotated video saved to 'output.mp4'")