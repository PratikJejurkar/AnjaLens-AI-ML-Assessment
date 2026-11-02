# Task 3 OCR Accuracy Report

## Runtime

- OCR engine: `easyocr`
- Python: `3.13.1 (tags/v3.13.1:0671451, Dec  3 2024, 19:06:28) [MSC v.1942 64 bit (AMD64)]`
- Platform: `Windows-11-10.0.26200-SP0`
- OpenCV: `5.0.0`
- Total images: `15`
- Runtime seconds: `110.24`

## Summary

| Mode | Field Accuracy incl. nulls | Field Accuracy non-null GT | Serial CER aggregate | Serial CER mean/image |
|---|---:|---:|---:|---:|
| Raw OCR | 50.67% | 17.50% | 0.7638 | 0.7242 |
| Preprocessed OCR | 49.33% | 15.00% | 0.9606 | 0.9099 |

## Field-Level Accuracy by Field

| Field | Raw incl. nulls | Preprocessed incl. nulls | Raw non-null GT | Preprocessed non-null GT |
|---|---:|---:|---:|---:|
| `equipment_id` | 53.33% | 53.33% | 0.00% | 0.00% |
| `serial_number` | 26.67% | 20.00% | 21.43% | 14.29% |
| `model` | 60.00% | 53.33% | 16.67% | 0.00% |
| `voltage` | 33.33% | 33.33% | 18.18% | 18.18% |
| `manufacture_date` | 80.00% | 86.67% | 25.00% | 50.00% |

## Notes

- `labels_output.json` contains the final structured output from the preprocessed OCR path.
- `labels_output_raw.json` contains extraction results without OpenCV preprocessing.
- Empty ground-truth values are treated as `null`. Accuracy is reported both including nulls and only for non-null ground-truth fields.
- Serial-number CER is computed only on rows where ground truth serial number is available.
