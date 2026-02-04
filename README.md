# vision, Student ID photo processing for TASS

A small Python CLI that batch-processes student photos into **TASS-ready ID images** by detecting the primary face, cropping with margin, resizing to a fixed target size, and compressing outputs to a strict file-size cap. If no suitable face is detected, it falls back to a centered, aspect-preserving crop.

## What’s in this repo

This repository includes everything required to run:

- `vision.py` (CLI entrypoint, includes shebang for direct execution)
- `deploy.prototxt` (OpenCV DNN face detector network definition)
- `res10_300x300_ssd_iter_140000.caffemodel` (OpenCV DNN face detector weights)

## Features

- **Face-aware crop** using OpenCV DNN (Caffe SSD face detector)
- **Deterministic output**: fixed target size, capped file size
- **Safe fallback**: centered crop if face detection fails
- **EXIF rotation correction** (common phone camera issue)
- **Quality checks + TASS warnings** (sharpness/brightness/contrast, etc.)
- **Machine-friendly reporting**: optional **JSONL** output, one record per input
- **Sequential processing by design** (predictable resource usage)

## Requirements

- Python 3
- OpenCV (`cv2`)
- NumPy
- Pillow (PIL)

Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install opencv-python numpy pillow
````

## Quick start

### 1) Make the script executable (optional)

`vision.py` includes a shebang, so you can run it directly:

```bash
chmod +x vision.py
```

### 2) Run (basic)

```bash
./vision.py input_folder/ output_folder/
# or
python3 vision.py input_folder/ output_folder/
```

By default, it only shows warnings/errors. Use `--log-level INFO` for detailed output.

## Usage

### Verbose output

```bash
./vision.py input_folder/ output_folder/ --log-level INFO
```

### Dry run (no files written)

```bash
./vision.py input_folder/ output_folder/ --dry-run --log-level INFO
```

### JSON Lines output (machine integration)

```bash
./vision.py input_folder/ output_folder/ --jsonl > results.jsonl
```

Example: list files that emitted TASS warnings

```bash
./vision.py input_folder/ output_folder/ --jsonl | \
  jq -r 'select(.tass_warnings | length > 0) | .input'
```

## CLI options

```text
--jsonl              Emit one JSON report per file to stdout
--no-summary         Suppress end-of-run summary line
--dry-run            Process without writing output files
--log-level LEVEL    DEBUG|INFO|WARNING|ERROR|CRITICAL (default: WARNING)
```

## Output expectations

* Output files are written into your specified `output_folder/`
* The tool aims to meet the configured target dimensions and file-size cap
* If a file cannot be processed, the JSONL record (if enabled) will capture the failure, and the process continues with the next file

## Customisation (advanced)

Processing parameters live in `DEFAULT_CONFIG` in `vision.py`. You can modify the defaults directly, or (if you’re importing this as a module) construct a custom config using `dataclasses.replace`.

Example pattern:

```python
from dataclasses import replace
from vision import DEFAULT_CONFIG, process_images_in_folder

custom_config = replace(
    DEFAULT_CONFIG,
    confidence_threshold=0.60,
    max_file_size_kb=40,
    prefer_upper_faces=False,
    sharpness_threshold=150.0,
)

process_images_in_folder(
    "input_folder",
    "output_folder",
    config=custom_config,
)
```

## Troubleshooting

### “Model files not found”

Confirm `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` are present in the same folder as `vision.py`.

### “No face detected” / poor crop

* Image may be low contrast, blurry, too small, or face is too far away.
* The tool will fall back to a centered crop when detection fails.

### OpenCV install issues

If `opencv-python` fails to install on your platform, consider:

* using your OS package manager, or
* using a virtual environment and a supported Python version.

## Exit codes

* `0` success (even if some files had warnings)
* non-zero indicates a fatal error (e.g., invalid args, missing required files, cannot read input folder)

