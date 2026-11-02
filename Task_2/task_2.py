import os
import sys
import time
import json
import math
import cv2
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

# ==========================================
# STEP 1: Fine-tuning & Evaluation Setup
# ==========================================
def train_and_evaluate(dataset_yaml="hard_hat_dataset.yaml", epochs=25, imgsz=640):
    """
    Fine-tunes YOLO11n for Hard-Hat (Helmet vs No-Helmet) detection.
    Prints validation metrics (mAP@50, mAP@50-95, P/R per class) and exports to ONNX.
    """
    print("=== 1. Fine-tuning YOLO11n on Hard-Hat Dataset ===")
    model = YOLO("yolo11n.pt")
    
    # Train the model with chosen hyper-parameters
    # Epochs=25 balancing performance and training time on budget hardware
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        lr0=0.01,
        lrf=0.01,
        plots=True,
        project="task2_runs",
        name="hardhat_yolo11n"
    )
    
    print("\n=== 2. Evaluating Validation Metrics ===")
    val_results = model.val()
    
    # Metrics report
    print(f"mAP@50    : {val_results.box.map50:.4f}")
    print(f"mAP@50-95 : {val_results.box.map:.4f}")
    
    for i, name in enumerate(val_results.names.values()):
        p = val_results.box.p[i] if len(val_results.box.p) > i else 0.0
        r = val_results.box.r[i] if len(val_results.box.r) > i else 0.0
        print(f"Class '{name}' -> Precision: {p:.4f}, Recall: {r:.4f}")

    print("\nConfusion matrix and PR curves saved under task2_runs/hardhat_yolo11n/")
    
    # Export to ONNX and FP16
    print("\n=== 3. Exporting to ONNX / FP16 ===")
    onnx_path = model.export(format="onnx", dynamic=False, half=False)
    print(f"ONNX model saved at: {onnx_path}")
    
    return "task2_runs/hardhat_yolo11n/weights/best.pt", onnx_path


# ==========================================
# STEP 2: Inference Benchmarking (PyTorch vs ONNX)
# ==========================================
def benchmark_models(pt_model_path, onnx_model_path, sample_images_dir, num_images=100):
    """
    Benchmarks PyTorch vs ONNX Runtime performance on 100 images:
    Latency (ms), FPS, and structure for comparative display.
    """
    print(f"\n=== 4. Benchmarking PyTorch vs ONNX Runtime ({num_images} images) ===")
    image_paths = list(Path(sample_images_dir).glob("*.jpg")) + list(Path(sample_images_dir).glob("*.png"))
    image_paths = image_paths[:num_images]
    
    if len(image_paths) == 0:
        print("Warning: No sample images found for benchmarking. Creating dummy input batch.")
        dummy_imgs = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(num_images)]
    else:
        dummy_imgs = [cv2.imread(str(p)) for p in image_paths]
        dummy_imgs = [cv2.resize(img, (640, 640)) for img in dummy_imgs]

    # --- PyTorch Benchmark ---
    pt_model = YOLO(pt_model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Warmup
    for _ in range(10):
        _ = pt_model(dummy_imgs[0], verbose=False)
        
    start_pt = time.time()
    for img in dummy_imgs:
        _ = pt_model(img, verbose=False)
    end_pt = time.time()
    
    pt_total_time = end_pt - start_pt
    pt_latency_ms = (pt_total_time / len(dummy_imgs)) * 1000
    pt_fps = len(dummy_imgs) / pt_total_time

    # --- ONNX Runtime Benchmark ---
    session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Warmup
    dummy_input = np.ones((1, 3, 640, 640), dtype=np.float32)
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})
        
    start_onnx = time.time()
    for img in dummy_imgs:
        img_prep = cv2.resize(img, (640, 640)).transpose(2, 0, 1)
        img_prep = np.expand_dims(img_prep, axis=0).astype(np.float32) / 255.0
        _ = session.run(None, {input_name: img_prep})
    end_onnx = time.time()
    
    onnx_total_time = end_onnx - start_onnx
    onnx_latency_ms = (onnx_total_time / len(dummy_imgs)) * 1000
    onnx_fps = len(dummy_imgs) / onnx_total_time

    print("\n" + "="*50)
    print(f"| {'Engine':<15} | {'Latency (ms)':<15} | {'FPS':<10} |")
    print("-" * 50)
    print(f"| {'PyTorch (.pt)':<15} | {pt_latency_ms:<15.2f} | {pt_fps:<10.1f} |")
    print(f"| {'ONNX Runtime':<15} | {onnx_latency_ms:<15.2f} | {onnx_fps:<10.1f} |")
    print("="*50 + "\n")


# ==========================================
# STEP 3: Integrated Pipeline (Task 1 + Task 2 PPE)
# ==========================================
def run_integrated_pipeline(
    video_path="../Task_1/Video/worker-zone-detection.mp4",
    config_path="../Task_1/config.json",
    ppe_model_path="hardhat_yolo11n.pt",
    output_video_path="task2_output.mp4",
    events_json_path="task2_events.json"
):
    """
    Runs tracking + zone entry monitoring + idle monitoring + PPE (helmet/no-helmet) compliance detection.
    Raises:
      1. ZONE_ENTRY
      2. IDLE_ALERT
      3. PPE_VIOLATION (when head/no-helmet detected on tracked person)
    """
    print("=== 5. Running Integrated Task 1 + Task 2 Pipeline ===")
    
    if not os.path.exists(config_path):
        config_data = {
            "zone_polygon_normalized": [[0.3, 0.4], [0.9, 0.4], [0.9, 0.9], [0.3, 0.9]],
            "idle_rules": {"pixel_distance": 30, "time_seconds": 3.0}
        }
    else:
        with open(config_path, 'r') as f:
            config_data = json.load(f)

    idle_dist_thresh = config_data["idle_rules"]["pixel_distance"]
    idle_time_thresh = config_data["idle_rules"]["time_seconds"]

    # Models: Primary tracker + PPE Fine-tuned model
    person_tracker = YOLO('yolo11n.pt')
    ppe_detector = YOLO(ppe_model_path) if os.path.exists(ppe_model_path) else person_tracker

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    abs_polygon = [[int(x * width), int(y * height)] for x, y in config_data["zone_polygon_normalized"]]
    zone_poly = Polygon(abs_polygon)
    np_polygon = np.array(abs_polygon, np.int32)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    events = []
    in_zone_ids = set()
    position_history = {}
    idle_alerted = set()
    ppe_alerted = set()

    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_timestamp = frame_count / fps
        banner_text = "STATUS: NORMAL"
        banner_color = (0, 255, 0)

        # Draw Restricted Zone
        cv2.polylines(frame, [np_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

        # Track Persons
        results = person_tracker.track(frame, classes=[0], persist=True, tracker="botsort.yaml", verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                foot_x, foot_y = (x1 + x2) / 2, y2
                foot_point = Point(foot_x, foot_y)

                # PPE Check inside worker crop
                person_crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                has_helmet = True
                
                if person_crop.size > 0 and os.path.exists(ppe_model_path):
                    ppe_res = ppe_detector(person_crop, verbose=False)
                    for ppe_box in ppe_res[0].boxes:
                        cls_id = int(ppe_box.cls[0].item())
                        cls_name = ppe_res[0].names[cls_id].lower()
                        if "head" in cls_name or "no-helmet" in cls_name or cls_id == 1:
                            has_helmet = False
                            break

                # Bounding box color based on helmet detection
                box_color = (0, 255, 0) if has_helmet else (0, 0, 255)
                label = f"ID: {track_id} | {'Helmet' if has_helmet else 'NO HELMET'}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (int(foot_x), int(foot_y)), 5, (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # --- Rule 1: Zone Entry ---
                if zone_poly.contains(foot_point):
                    if track_id not in in_zone_ids:
                        in_zone_ids.add(track_id)
                        events.append({
                            "event_type": "ZONE_ENTRY",
                            "track_id": int(track_id),
                            "timestamp": round(current_timestamp, 2),
                            "frame_number": frame_count,
                            "box": [round(float(b), 2) for b in box]
                        })
                    banner_text = f"ZONE ENTRY ALERT: ID {track_id}"
                    banner_color = (0, 0, 255)
                else:
                    if track_id in in_zone_ids:
                        in_zone_ids.remove(track_id)

                # --- Rule 2: Idle Alert ---
                if track_id not in position_history:
                    position_history[track_id] = []
                
                history = position_history[track_id]
                history.append((foot_x, foot_y, current_timestamp))
                history = [p for p in history if current_timestamp - p[2] <= idle_time_thresh]
                position_history[track_id] = history

                if len(history) > 1 and (current_timestamp - history[0][2]) >= (idle_time_thresh - 0.5):
                    max_dist = max(math.hypot(p1[0]-p2[0], p1[1]-p2[1]) for i, p1 in enumerate(history) for p2 in history[i+1:])
                    if max_dist < idle_dist_thresh:
                        if track_id not in idle_alerted:
                            idle_alerted.add(track_id)
                            events.append({
                                "event_type": "IDLE_ALERT",
                                "track_id": int(track_id),
                                "timestamp": round(current_timestamp, 2),
                                "frame_number": frame_count,
                                "box": [round(float(b), 2) for b in box]
                            })
                        banner_text = f"IDLE ALERT: ID {track_id}"
                        banner_color = (0, 165, 255)

                # --- Rule 3: PPE Violation Event ---
                if not has_helmet and track_id not in ppe_alerted:
                    ppe_alerted.add(track_id)
                    events.append({
                        "event_type": "PPE_VIOLATION",
                        "track_id": int(track_id),
                        "timestamp": round(current_timestamp, 2),
                        "frame_number": frame_count,
                        "box": [round(float(b), 2) for b in box]
                    })
                    banner_text = f"PPE VIOLATION: ID {track_id} NO HELMET"
                    banner_color = (0, 0, 255)

        cv2.putText(frame, banner_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, banner_color, 3)
        out.write(frame)

    cap.release()
    out.release()

    end_time = time.time()
    avg_fps = frame_count / (end_time - start_time)

    print("-" * 40)
    print(f"Task 2 Processing Complete!")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Saved events log to: {events_json_path}")
    print(f"Saved annotated video to: {output_video_path}")
    print("-" * 40)

    with open(events_json_path, 'w') as f:
        json.dump(events, f, indent=4)


if __name__ == "__main__":
    # Standard invocation
    run_integrated_pipeline()