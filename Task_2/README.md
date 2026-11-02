# Task 2 – Integrated Worker Monitoring and PPE Compliance

## 1. Objective

Extend the worker-monitoring pipeline with PPE/helmet checks. The integrated pipeline tracks people, monitors restricted-zone entry and inactivity, checks each person crop for helmet-related detections when a trained PPE model is available, annotates the video, and writes event JSON.

The script also contains separate functions for hard-hat model training/evaluation and PyTorch-versus-ONNX benchmarking.

## 2. Technology

- Python, OpenCV, NumPy
- Ultralytics YOLO11n and BoT-SORT
- PyTorch for model inference and device selection
- ONNX Runtime for exported-model benchmarking
- Shapely for restricted-zone geometry
- JSON for configuration and event records

## 3. Inputs and dependencies

- Input video: `../Task_1/Video/worker-zone-detection.mp4` (relative to `Task_2`)
- Zone/idle configuration: `../Task_1/config.json`; defaults are used if not found
- Person detector/tracker weights: `yolo11n.pt`
- PPE weights: `hardhat_yolo11n.pt` (must be trained/provided for actual helmet detection)
- Optional training data: a YOLO dataset YAML such as `hard_hat_dataset.yaml` and its referenced images/labels

The assessment archive does not include the hard-hat dataset YAML or the trained `hardhat_yolo11n.pt` file.

## 4. Integrated processing workflow

1. Load the Task 1 zone and idle configuration, or use built-in defaults.
2. Initialize YOLO11n for person detection and BoT-SORT tracking.
3. Initialize the PPE model if `hardhat_yolo11n.pt` exists.
4. Read each video frame and obtain persistent person track IDs.
5. Crop each tracked person and run the PPE model on the crop.
6. Apply zone-entry logic using the bottom-center point of the person's bounding box.
7. Apply rolling-window idle logic using the configured time and pixel-distance thresholds.
8. Emit a `PPE_VIOLATION` event when the PPE model indicates a no-helmet/head-related class.
9. Annotate boxes, labels, zone polygon and alert banner; save the video and JSON event list.

## 5. Event types

- `ZONE_ENTRY`
- `IDLE_ALERT`
- `PPE_VIOLATION`

Event fields are `event_type`, `track_id`, `timestamp`, `frame_number`, and `box`.

## 6. Run (Windows CMD)

From the repository root:

```bat
cd Task_2
python task_2.py
```

The default output files are:

- `task2_output.mp4`
- `task2_events.json`

## 7. Training and benchmarking functions

`train_and_evaluate(dataset_yaml="hard_hat_dataset.yaml", epochs=25, imgsz=640)` fine-tunes YOLO11n, prints validation metrics, and exports an ONNX model. The dataset YAML and training images must be supplied separately.

`benchmark_models(pt_model_path, onnx_model_path, sample_images_dir, num_images=100)` compares PyTorch and ONNX Runtime inference latency/FPS using sample images or generated dummy images if none are found. These functions are defined in the script but are not automatically executed by its current `__main__` block.

## 8. Important PPE fallback behavior

If `hardhat_yolo11n.pt` is missing, the current code sets the PPE detector to the person tracker, but the PPE check is only executed when the PPE weights file exists. As a result, `has_helmet` remains `True` and no reliable PPE violations can be detected. A valid PPE model with documented class names and training/validation results is required before interpreting the output as helmet compliance.

## 9. Limitations and next improvements

- Confirm PPE class mapping against the trained model's actual labels; current logic treats class ID `1` or labels containing `head`/`no-helmet` as a violation.
- Evaluate person-crop detection under small, distant and occluded subjects.
- Track per-person PPE state with confidence thresholds and temporal voting to reduce flicker.
- Add event cooldown/re-entry rules and persistent event storage.
- Add CLI options for video, weights, config and output paths.
- Include training dataset provenance, class balance, validation metrics and model version.
- Benchmark on the target deployment hardware with warm-up, repeated runs and identical input sets.
