# Task 1 – Worker Zone Entry and Idle Monitoring

## 1. Objective

Process a factory-floor video to detect and track people, identify entry into a configured restricted polygon, detect prolonged inactivity, annotate the video, and save structured event records.

## 2. Technology

- Python
- OpenCV for video input/output, drawing and frame handling
- Ultralytics YOLO11n for person detection and BoT-SORT tracking
- PyTorch for CPU/GPU selection
- Shapely for point-in-polygon checks
- NumPy for polygon coordinates
- JSON for configuration and event output

## 3. Inputs

- Video: `Video/worker-zone-detection.mp4` in the supplied archive
- Configuration: `config.json`
- YOLO weights: `yolo11n.pt` (downloaded automatically by Ultralytics if not present, subject to network access)

The configuration contains normalized polygon vertices and idle thresholds:

```json
{
  "zone_polygon_normalized": [[0.3, 0.4], [0.9, 0.4], [0.9, 0.9], [0.3, 0.9]],
  "idle_rules": {
    "pixel_distance": 30,
    "time_seconds": 3.0
  }
}
```

Normalized coordinates are scaled to the input video's width and height. The person's bottom-center point (foot point) is used for zone membership.

## 4. Processing workflow

1. Download the sample video if the expected local video file does not exist.
2. Write/load the JSON configuration.
3. Initialize YOLO11n and select CUDA when available, otherwise CPU.
4. Read video frames and track class `0` (person) using BoT-SORT with persistent track IDs.
5. Convert the configured normalized polygon to pixel coordinates.
6. Test each tracked person's foot point against the restricted polygon.
7. Maintain a rolling position history for each track. If movement stays below the configured pixel-distance threshold over the configured time window, emit an idle event.
8. Draw the polygon, bounding boxes, track IDs, foot points and alert banner.
9. Write the annotated video and event records.

## 5. Event schema

The script emits `ZONE_ENTRY` and `IDLE_ALERT` events. Each event contains:

```json
{
  "event_type": "ZONE_ENTRY",
  "track_id": 2,
  "timestamp": 4.99,
  "frame_number": 299,
  "box": [582.22, 232.64, 837.87, 965.25]
}
```

`timestamp` is video time in seconds; `frame_number` is the processed frame index; `box` is `[x1, y1, x2, y2]`.

## 6. Run (Windows CMD)

From the repository root:

```bat
cd Task_1
mkdir video
copy Video\worker-zone-detection.mp4 video\worker-zone-detection.mp4
python task_1.py
```

The lowercase `video` directory is required by the current script path. The archive stores the source clip in uppercase `Video`. Alternatively, change `video_path` in `task_1.py` to `Video/worker-zone-detection.mp4`.

## 7. Outputs

Created in the current working directory (`Task_1/`):

- `output.mp4` – annotated video
- `events.json` – detected events
- `config.json` – configuration (the script rewrites it with its in-code defaults)

The script prints the selected hardware and average processing FPS.

## 8. Limitations and next improvements

- The script uses fixed image-space pixel thresholds for inactivity; perspective and camera movement can affect the meaning of 30 pixels.
- Track IDs can be lost or reassigned during occlusion.
- `in_zone_ids` and `idle_alerted` are in-memory sets; events are not persisted incrementally during processing.
- The current implementation writes a default configuration each run, so manual edits to `config.json` are overwritten. Move default creation behind a file-exists check to preserve operator configuration.
- Add structured logging, input validation, configurable CLI arguments, confidence thresholds, camera calibration, event de-duplication policy, and automated tests before production use.
