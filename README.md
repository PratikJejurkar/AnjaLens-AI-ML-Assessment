# AnjaLens – AI, Computer Vision and OCR Assessment

This repository contains four independent practical tasks covering video-based worker monitoring, PPE compliance, equipment-label OCR, and a retrieval-augmented maintenance assistant.

## Tasks at a glance

| Task | Purpose | Main technologies | Main outputs |
|---|---|---|---|
| Task 1 | Detect people, restricted-zone entry and prolonged inactivity in video | Python, OpenCV, Ultralytics YOLO11n, BoT-SORT, Shapely | Annotated video and event JSON |
| Task 2 | Combine worker tracking, zone/idle monitoring and helmet/PPE violation detection; includes training and ONNX benchmarking functions | Python, OpenCV, YOLO11n, PyTorch, ONNX Runtime, Shapely | Annotated video and event JSON; optional training/benchmark artifacts |
| Task 3 | Extract structured equipment/nameplate fields from images and evaluate against ground truth | Python, OpenCV, EasyOCR/Tesseract, NumPy | JSON predictions and accuracy reports |
| Task 4 | Answer maintenance questions using manuals and retrieval-augmented generation, with simple equipment tools | FastAPI, LangChain, Chroma, Hugging Face embeddings/LLM | REST API responses, citations and inspection log |

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── Task_1/
│   ├── task_1.py
│   ├── config.json
│   ├── events.json
│   └── Video/
├── Task_2/
│   ├── task_2.py
│   └── task2_events.json
├── Task_3/
│   ├── data/
│   │   ├── Image_*.png
│   │   └── labels_ground_truth.csv
│   ├── src/run_ocr_task3.py
│   └── outputs/
├── Task_4/
│   ├── app/
│   │   ├── main.py
│   │   ├── tools.py
│   │   └── utils.py
│   ├── data/
│   ├── questions.json
│   └── requirements.txt
└── docs/
    └── System_Design_Document.md
```

## Environment setup (Windows CMD)

Use Python 3.13.1 to match the recorded Task 3 run, subject to package compatibility on your machine. A separate virtual environment is recommended.

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some ML packages, particularly PyTorch, may require a platform-specific installation. If installation fails, install the compatible PyTorch build for your CPU/GPU from the official PyTorch selector, then install the remaining requirements.

## Run each task

Run commands from the repository root unless noted otherwise.

### Task 1 – Worker zone and idle monitoring

The script uses a relative input path of `video/worker-zone-detection.mp4` (lowercase `video`) and writes `output.mp4`, `events.json`, and `config.json` relative to the current working directory. The supplied archive has the input video in `Task_1/Video/` (uppercase `Video`). To run without changing the script, create the expected lowercase folder and copy the video:

```bat
cd Task_1
mkdir video
copy Video\worker-zone-detection.mp4 video\worker-zone-detection.mp4
python task_1.py
```

### Task 2 – Integrated worker monitoring and PPE

The script's default input path is `../Task_1/Video/worker-zone-detection.mp4`, so run it from inside `Task_2`:

```bat
cd Task_2
python task_2.py
```

The integrated pipeline looks for `hardhat_yolo11n.pt`. If that trained PPE model is absent, the current code falls back to the general person model and does not provide meaningful helmet classification. A hard-hat training dataset YAML and its images are also not included in this repository archive; the training function requires those external dataset files.

### Task 3 – OCR

The supplied images are directly inside `Task_3/data/`, not a nested `images` folder. Run from the repository root:

```bat
python Task_3\src\run_ocr_task3.py --images_dir Task_3\data --ground_truth Task_3\data\labels_ground_truth.csv --out_dir Task_3\outputs --engine easyocr
```

To see all command-line options:

```bat
python Task_3\src\run_ocr_task3.py --help
```

### Task 4 – Maintenance RAG API

Create `Task_4/.env` locally and set your own Hugging Face token. Do not commit this file.

```text
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

Then run:

```bat
cd Task_4
uvicorn app.main:app --reload
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs` and health endpoint at `http://127.0.0.1:8000/health`.

## Task 4 API

### `GET /health`

Returns service health flags, including whether the LLM and vector store initialized.

### `POST /ask`

Example request:

```json
{
  "question": "What is the maximum pressure for the Hydraulic Pump Model X-200?"
}
```

The response contains `answer`, `citations`, and `tool_calls`.

## Important implementation notes

- Task 1 currently expects a lowercase `video` directory, while the provided input folder is named `Video`. Follow the run instructions above or update the script path consistently.
- Task 2's PPE classification depends on a trained `hardhat_yolo11n.pt` model. Without it, the script's fallback is not a valid PPE detector.
- Task 2's optional training routine requires an external hard-hat dataset YAML and images, which are not included.
- Task 3's existing evaluation report is a baseline from the included 15-image set. It reports raw field accuracy of 50.67% including null ground-truth fields and preprocessed accuracy of 49.33% including nulls; non-null-only accuracy is lower. Review `Task_3/outputs/accuracy_report.md` before interpreting performance.
- Task 4 uses a mock equipment-status function and JSON-file inspection logging, not a live PLC/CMMS/database integration.
- Task 4 requires a Hugging Face token for generated answers. Never commit API tokens, `.env` files, or credentials.
- The included Chroma database is generated local state. It can be rebuilt from the Markdown manuals by the application; do not treat it as the authoritative source of manuals.

## Documentation

- [Task 1 README](Task_1/README.md)
- [Task 2 README](Task_2/README.md)
- [Task 3 README](Task_3/README.md)
- [Task 4 README](Task_4/README.md)
- [System Design Document](docs/System_Design_Document.md)
