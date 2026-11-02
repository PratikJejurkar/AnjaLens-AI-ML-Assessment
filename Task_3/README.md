# Task 3 – Equipment Nameplate OCR and Structured Extraction

## 1. Objective

Read equipment/nameplate images, extract selected fields into structured records, and evaluate predictions against a supplied ground-truth CSV. The script supports EasyOCR and Tesseract engines and compares a raw-image OCR path with an OpenCV-preprocessed path.

## 2. Technology

- Python
- OpenCV and NumPy for image handling/preprocessing
- EasyOCR or Tesseract for text recognition
- CSV and JSON for inputs and outputs
- Python logging and runtime metadata for evaluation reporting

## 3. Input data

The supplied archive contains 15 images directly in `data/` and the label file `data/labels_ground_truth.csv`.

Fields evaluated by the report:

- `equipment_id`
- `serial_number`
- `model`
- `voltage`
- `manufacture_date`

## 4. Processing workflow

1. Parse command-line arguments and discover image files.
2. Load ground-truth rows from the CSV.
3. Initialize the selected OCR engine.
4. Run OCR on the original image (raw path).
5. Apply image preprocessing and run OCR again (preprocessed path).
6. Extract/normalize candidate values for the target fields.
7. Match image-level predictions to ground truth.
8. Calculate field-level exact-match accuracy, including-null and non-null-ground-truth views, plus serial-number character error rate (CER).
9. Write structured predictions, reports, OCR text, preprocessed images and logs to the output directory.

## 5. Run (Windows CMD)

From the repository root (images are directly in `Task_3\data`):

```bat
python Task_3\src\run_ocr_task3.py --images_dir Task_3\data --ground_truth Task_3\data\labels_ground_truth.csv --out_dir Task_3\outputs --engine easyocr
```

For all supported options:

```bat
python Task_3\src\run_ocr_task3.py --help
```

To use Tesseract instead:

```bat
python Task_3\src\run_ocr_task3.py --images_dir Task_3\data --ground_truth Task_3\data\labels_ground_truth.csv --out_dir Task_3\outputs --engine tesseract
```

Tesseract requires the Tesseract executable to be installed and available on PATH, in addition to the Python package dependencies.

## 6. Outputs

The script writes the following files under the selected output directory:

- `labels_output.json` – structured predictions from the preprocessed OCR path
- `labels_output_raw.json` – structured predictions from the raw OCR path
- `accuracy_report.json` – machine-readable metrics and runtime details
- `accuracy_report.md` – human-readable accuracy report
- `ocr_text/raw/` – raw OCR text per image
- `ocr_text/preprocessed/` – preprocessed OCR text per image
- `preprocessed/` – preprocessed image files
- `task3_ocr.log` – execution log

## 7. Recorded baseline (included report)

The included `outputs/accuracy_report.md` records a run on 15 images using EasyOCR:

| Metric | Raw OCR | Preprocessed OCR |
|---|---:|---:|
| Field accuracy including null ground truth | 50.67% | 49.33% |
| Field accuracy for non-null ground truth | 17.50% | 15.00% |
| Aggregate serial-number CER | 0.7638 | 0.9606 |

These are the results from the included run, not a guarantee of performance on other images. In this report, preprocessing did not improve aggregate field accuracy or serial-number CER. Exact-match field accuracy is strict and can penalize small OCR character differences.

## 8. Limitations and next improvements

- The baseline indicates that field extraction and normalization need further tuning; OCR text recognition alone does not guarantee correct structured values.
- Improve region-of-interest detection, perspective correction, denoising, contrast enhancement and adaptive threshold selection.
- Use field-specific validators (e.g. voltage patterns, serial formats, date parsing) and confidence-aware candidate ranking.
- Add per-field confusion/error examples and separate OCR recognition errors from extraction/normalization errors.
- Expand and stratify the evaluation set; report confidence intervals and per-image results.
- Validate image-to-ground-truth filename matching and missing-value policy.
