#!/opt/homebrew/bin/python3
"""
Image Processing Script - Center and Resize Images using Face Detection
=======================================================================

Purpose
-------
Prepare student ID photos for ingestion into a student management system by:
- detecting a face (primary subject),
- cropping around it with a margin,
- resizing to a fixed target size,
- saving to a constrained file size.

If no suitable face is detected, the script falls back to a centered aspect-preserving crop
to the target size.

Dependencies
------------
- OpenCV (cv2)
- NumPy (np)
- Pillow (PIL)

Model Files
-----------
Ensure the following model files are present in the same directory as this script:
1. res10_300x300_ssd_iter_140000.caffemodel
2. deploy.prototxt

Usage
-----
    python vision_hardened_v4.py <input_folder> <output_folder>
    
    Options:
        --jsonl              Emit JSON reports to stdout
        --no-summary         Suppress summary output
        --dry-run            Process without writing files
        --log-level LEVEL    Logging level (default: WARNING)
"""
from __future__ import annotations

import argparse
import os
import json
import sys
import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class ProcessingConfig:
    """Configurable processing parameters."""
    target_size: Tuple[int, int] = (500, 625)          # (width, height)
    margin: float = 0.5                                # margin around detected face
    max_file_size_kb: int = 50                         # output cap
    dpi: Tuple[int, int] = (72, 72)                    # output DPI metadata
    confidence_threshold: float = 0.50                 # minimum face confidence
    output_format: str = "JPEG"                        # deterministic output format
    prefer_upper_faces: bool = True                    # prefer faces in upper 2/3
    max_quality: int = 95                              # JPEG quality ceiling
    min_quality: int = 25                              # JPEG quality floor
    min_dim_floor: int = 250                           # minimum dimension after downscaling
    downscale_factor: float = 0.92                     # gentle downscale step
    sharpness_threshold: float = 100.0                 # blur detection threshold
    brightness_dark_threshold: float = 60.0            # darkness threshold
    min_image_size: int = 100                          # minimum input dimensions
    max_image_size: int = 10000                        # maximum input dimensions


DEFAULT_CONFIG = ProcessingConfig()


# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger(__name__)


# -----------------------------
# Model loading
# -----------------------------
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
MODEL_FILE = SCRIPT_DIRECTORY / "res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_FILE = SCRIPT_DIRECTORY / "deploy.prototxt"


def load_face_net() -> cv2.dnn_Net:
    """Load the OpenCV DNN face detection model with explicit errors."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing model config: {CONFIG_FILE}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing model weights: {MODEL_FILE}")
    return cv2.dnn.readNetFromCaffe(str(CONFIG_FILE), str(MODEL_FILE))


# -----------------------------
# Image validation
# -----------------------------
def validate_image(image: Image.Image, config: ProcessingConfig) -> None:
    """Ensure image is processable."""
    if image.width < config.min_image_size or image.height < config.min_image_size:
        raise ValueError(f"Image too small: {image.size} (minimum {config.min_image_size}x{config.min_image_size})")
    if image.width > config.max_image_size or image.height > config.max_image_size:
        raise ValueError(f"Image unreasonably large: {image.size} (maximum {config.max_image_size}x{config.max_image_size})")


# -----------------------------
# Image IO helpers
# -----------------------------
def open_image_exif_corrected(image_path: str, config: ProcessingConfig) -> Image.Image:
    """Open image via PIL and apply EXIF orientation correction (common for phones)."""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    
    # Normalize to RGB for consistent downstream behavior.
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Validate dimensions
    validate_image(img, config)
    
    return img


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR ndarray."""
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# -----------------------------
# Image quality assessment
# -----------------------------
def assess_quality(image: Image.Image, config: ProcessingConfig) -> dict:
    """Return quality metrics for debugging and validation."""
    img_array = np.array(image)
    
    # Sharpness (Laplacian variance)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Brightness
    brightness = float(np.mean(img_array))
    
    # Contrast (standard deviation)
    contrast = float(np.std(img_array))
    
    return {
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "is_blurry": sharpness < config.sharpness_threshold,
        "is_dark": brightness < config.brightness_dark_threshold,
    }


# -----------------------------
# Face detection and selection
# -----------------------------
@dataclass(frozen=True)
class FaceCandidate:
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
    
    @property
    def center_y(self) -> int:
        return (self.y1 + self.y2) // 2


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    """Clamp coordinates into image bounds and ensure x1<x2, y1<y2."""
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def detect_faces(face_net: cv2.dnn_Net, image_bgr: np.ndarray, config: ProcessingConfig) -> List[FaceCandidate]:
    """Return face candidates above threshold."""
    h, w = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, (300, 300))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    candidates: List[FaceCandidate] = []
    # OpenCV returns a fixed tensor with N detections; filter by confidence.
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < config.confidence_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int).tolist()
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        # Skip degenerate boxes.
        if (x2 - x1) <= 1 or (y2 - y1) <= 1:
            continue
        candidates.append(FaceCandidate(confidence, x1, y1, x2, y2))
    return candidates


def choose_primary_face(
    candidates: List[FaceCandidate], 
    image_height: int,
    config: ProcessingConfig
) -> Optional[FaceCandidate]:
    """
    Pick the primary subject face.
    Enhanced heuristic for student ID photos:
    - Prefer larger faces (likely the subject)
    - Prefer faces in upper 2/3 of image (typical framing)
    - Use confidence as tiebreaker
    """
    if not candidates:
        return None
    
    if config.prefer_upper_faces:
        # Weight faces in upper portion (students typically centered high in ID photos)
        def score(c: FaceCandidate) -> Tuple[float, float, float]:
            # Vertical bias: full weight if in upper 2/3, 80% if lower
            vertical_bias = 1.0 if c.center_y < image_height * 0.66 else 0.8
            # Sort by: (area * bias, confidence, prefer higher in frame)
            return (c.area * vertical_bias, c.confidence, -c.y1)
        return max(candidates, key=score)
    
    return max(candidates, key=lambda c: (c.area, c.confidence))


# -----------------------------
# Resizing, cropping, saving
# -----------------------------
def resize_and_center_crop(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Resize and center-crop to the target size while preserving aspect ratio."""
    tw, th = target_size
    aspect_ratio = image.width / image.height
    target_ratio = tw / th

    if aspect_ratio > target_ratio:
        # Image is wider than target: fit height, crop width
        new_height = th
        new_width = int(round(new_height * aspect_ratio))
    else:
        # Image is taller than target: fit width, crop height
        new_width = tw
        new_height = int(round(new_width / aspect_ratio))

    image = image.resize((new_width, new_height), Image.LANCZOS)
    left = max(0, (new_width - tw) // 2)
    top = max(0, (new_height - th) // 2)
    return image.crop((left, top, left + tw, top + th))


def crop_around_face(image_bgr: np.ndarray, face: FaceCandidate, margin: float) -> np.ndarray:
    """Crop around the face with margin, clamped to image bounds."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = face.x1, face.y1, face.x2, face.y2
    face_w = x2 - x1
    face_h = y2 - y1

    # Expand box by margin around the face.
    mx = int(round(margin * face_w))
    my = int(round(margin * face_h))

    cx = x1 + face_w // 2
    cy = y1 + face_h // 2

    crop_x1 = cx - (face_w // 2) - mx
    crop_y1 = cy - (face_h // 2) - my
    crop_x2 = cx + (face_w // 2) + mx
    crop_y2 = cy + (face_h // 2) + my

    crop_x1, crop_y1, crop_x2, crop_y2 = clamp_box(crop_x1, crop_y1, crop_x2, crop_y2, w, h)
    # Ensure non-empty crop.
    if (crop_x2 - crop_x1) <= 1 or (crop_y2 - crop_y1) <= 1:
        return image_bgr
    return image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]


# -----------------------------
# Smart size estimation and compression
# -----------------------------
def estimate_jpeg_size_fast(image: Image.Image, quality: int, config: ProcessingConfig) -> int:
    """Fast in-memory JPEG size estimation without disk I/O."""
    buffer = BytesIO()
    image.save(
        buffer,
        format=config.output_format,
        quality=int(quality),
        optimize=True,
        progressive=True,
    )
    return buffer.tell()


def estimate_initial_scale(image: Image.Image, target_bytes: int) -> Image.Image:
    """Pre-emptively downscale if image is obviously too large."""
    # Rough heuristic: JPEG compression ~0.15-0.25 bytes/pixel at medium quality
    current_pixels = image.width * image.height
    estimated_bytes = current_pixels * 0.2  # conservative estimate
    
    if estimated_bytes > target_bytes * 1.5:
        # Need to downscale
        scale = min(0.95, (target_bytes / estimated_bytes) ** 0.5)
        new_w = max(250, int(image.width * scale))
        new_h = max(250, int(image.height * scale))
        logger.debug(f"Pre-emptive downscale: {image.size} -> ({new_w}, {new_h})")
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image


def compress_and_save_jpeg(
    image: Image.Image,
    output_path: str,
    config: ProcessingConfig,
    dry_run: bool = False,
) -> dict:
    """
    Save deterministically as JPEG and attempt to meet size cap by:
    1) Pre-emptive downscaling if obviously too large
    2) Finding the highest JPEG quality that meets the cap (binary search in memory)
    3) If still oversize at min_quality, progressively downscaling and retrying
    
    Note: In rare cases where the image cannot be compressed below the target without
    going below the minimum dimension floor, the status will be "OVERSIZE".
    
    Returns metadata for machine-readable reporting.
    """
    if not dry_run:
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = int(config.max_file_size_kb) * 1024
    
    # Pre-emptive downscaling to avoid unnecessary work
    working = estimate_initial_scale(image, target_bytes)
    scale_steps = 0 if working.size == image.size else 1

    def find_best_quality(img: Image.Image) -> Tuple[int, int, BytesIO]:
        """Return (best_quality, best_size_bytes, buffer) where best_quality meets cap if possible."""
        # If even max_quality meets cap, we're done.
        buffer = BytesIO()
        img.save(
            buffer,
            format=config.output_format,
            quality=config.max_quality,
            optimize=True,
            progressive=True,
            dpi=config.dpi,
        )
        size_at_max = buffer.tell()
        
        if size_at_max <= target_bytes:
            return config.max_quality, size_at_max, buffer

        # If even min_quality is too large, signal failure to caller.
        size_at_min = estimate_jpeg_size_fast(img, config.min_quality, config)
        if size_at_min > target_bytes:
            # Return min_quality buffer for final save
            buffer_min = BytesIO()
            img.save(
                buffer_min,
                format=config.output_format,
                quality=config.min_quality,
                optimize=True,
                progressive=True,
                dpi=config.dpi,
            )
            return config.min_quality, buffer_min.tell(), buffer_min

        # Binary search for highest quality that still meets cap.
        lo, hi = config.min_quality, config.max_quality
        best_q, best_size = config.min_quality, size_at_min
        best_buffer = None
        
        while lo <= hi:
            mid = (lo + hi) // 2
            test_buffer = BytesIO()
            img.save(
                test_buffer,
                format=config.output_format,
                quality=mid,
                optimize=True,
                progressive=True,
                dpi=config.dpi,
            )
            size_mid = test_buffer.tell()
            
            if size_mid <= target_bytes:
                best_q, best_size = mid, size_mid
                best_buffer = test_buffer
                lo = mid + 1
            else:
                hi = mid - 1
        
        return best_q, best_size, best_buffer

    best_quality, best_size, buffer = find_best_quality(working)

    # If oversize at min_quality, downscale until it fits or we hit the floor.
    while best_size > target_bytes:
        w, h = working.size
        # Stop if we'd go below floor.
        if min(w, h) <= config.min_dim_floor:
            logger.warning(f"Cannot compress below {config.max_file_size_kb}KB without going below {config.min_dim_floor}px floor")
            break

        # Downscale gently; keep aspect ratio.
        nw = max(config.min_dim_floor, int(round(w * config.downscale_factor)))
        nh = max(config.min_dim_floor, int(round(h * config.downscale_factor)))
        if (nw, nh) == (w, h):
            break

        working = working.resize((nw, nh), Image.LANCZOS)
        scale_steps += 1
        best_quality, best_size, buffer = find_best_quality(working)
        logger.debug(f"Downscale step {scale_steps}: {working.size}, quality={best_quality}, size={best_size}")

    # Determine status
    if best_size > target_bytes:
        status = "OVERSIZE"
        logger.warning(f"File still oversize: {best_size} bytes > {target_bytes} bytes")
    else:
        status = "OK"

    # Write final file
    if not dry_run:
        with open(output_path, 'wb') as f:
            f.write(buffer.getvalue())

    return {
        "status": status,
        "bytes": int(best_size),
        "target_bytes": int(target_bytes),
        "quality": int(best_quality),
        "width": int(working.size[0]),
        "height": int(working.size[1]),
        "downscale_steps": int(scale_steps),
    }


# -----------------------------
# TASS validation
# -----------------------------
def validate_for_tass(report: dict, config: ProcessingConfig) -> List[str]:
    """Return list of warnings for TASS compatibility."""
    warnings = []
    
    if report.get("status") == "OVERSIZE":
        warnings.append(f"File exceeds TASS size limit ({config.max_file_size_kb}KB)")
    
    quality_data = report.get("quality_metrics", {})
    if quality_data.get("is_blurry"):
        warnings.append(f"Image may be too blurry for ID card (sharpness: {quality_data.get('sharpness', 0)})")
    
    if quality_data.get("is_dark"):
        warnings.append(f"Image may be too dark (brightness: {quality_data.get('brightness', 0)})")
    
    if report.get("face") is None:
        warnings.append("No face detected - manual review recommended")
    
    if report.get("downscale_steps", 0) > 3:
        warnings.append(f"Significant downscaling applied ({report['downscale_steps']} steps) - check image quality")
    
    face_confidence = report.get("face", {}).get("confidence", 0) if report.get("face") else 0
    if face_confidence > 0 and face_confidence < 0.7:
        warnings.append(f"Low face detection confidence ({face_confidence:.2f}) - verify result")
    
    return warnings


# -----------------------------
# Output path generation
# -----------------------------
def output_path_for_input(output_folder: str, input_filename: str) -> str:
    """
    Generate output path. Enforce JPEG extension for deterministic size control.
    Keeps basename, normalizes extension to .jpg.
    """
    in_path = Path(input_filename)
    base = in_path.stem
    return str(Path(output_folder) / f"{base}.jpg")


# -----------------------------
# Image processing
# -----------------------------
def process_image_without_face(
    image_pil: Image.Image, 
    output_path: str, 
    config: ProcessingConfig,
    dry_run: bool = False,
) -> dict:
    """Process image without face detection (fallback path)."""
    resized = resize_and_center_crop(image_pil, config.target_size)
    meta = compress_and_save_jpeg(resized, output_path, config, dry_run)
    
    # Assess quality on final image
    quality_metrics = assess_quality(resized, config)
    meta["quality_metrics"] = quality_metrics
    
    return meta


def process_single_image(
    face_net: cv2.dnn_Net, 
    image_path: str, 
    output_path: str,
    config: ProcessingConfig = DEFAULT_CONFIG,
    dry_run: bool = False,
) -> Optional[dict]:
    """
    Process a single image. Returns a machine-readable report dict on success or soft-failure.
    Returns None only if the file could not be opened at all.
    """
    try:
        image_pil = open_image_exif_corrected(image_path, config)
    except ValueError as e:
        logger.error(f"Validation error for {image_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unable to open image {image_path}: {e}")
        return None

    image_bgr = pil_to_bgr(image_pil)

    face: Optional[FaceCandidate]
    face_error: Optional[str] = None
    try:
        candidates = detect_faces(face_net, image_bgr, config)
        face = choose_primary_face(candidates, image_bgr.shape[0], config)
        if face:
            logger.debug(f"Detected face with confidence {face.confidence:.3f} at {[face.x1, face.y1, face.x2, face.y2]}")
    except Exception as e:
        # Treat detector errors as 'no face' fallback, but preserve error context in report.
        logger.warning(f"Face detection failed for {image_path}: {e}. Falling back to center crop.")
        face = None
        face_error = str(e)

    if face is None:
        meta = process_image_without_face(image_pil, output_path, config, dry_run)
        report = {
            "input": image_path,
            "output": output_path,
            "face": None,
            "face_error": face_error,
            **meta,
        }
        # Add TASS warnings
        report["tass_warnings"] = validate_for_tass(report, config)
        
        # Cleanup
        del image_pil, image_bgr
        gc.collect()
        
        return report

    # Face detected - crop and process
    cropped_bgr = crop_around_face(image_bgr, face, config.margin)
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))
    cropped_resized = resize_and_center_crop(cropped_pil, config.target_size)
    meta = compress_and_save_jpeg(cropped_resized, output_path, config, dry_run)
    
    # Assess quality on final image
    quality_metrics = assess_quality(cropped_resized, config)
    meta["quality_metrics"] = quality_metrics
    
    report = {
        "input": image_path,
        "output": output_path,
        "face": {
            "confidence": round(face.confidence, 4), 
            "box": [face.x1, face.y1, face.x2, face.y2]
        },
        "face_error": None,
        **meta,
    }
    
    # Add TASS warnings
    report["tass_warnings"] = validate_for_tass(report, config)
    
    # Cleanup
    del image_pil, image_bgr, cropped_bgr, cropped_pil, cropped_resized
    gc.collect()
    
    return report


# -----------------------------
# Batch processing
# -----------------------------
def process_images_in_folder(
    input_folder: str,
    output_folder: str,
    *,
    emit_jsonl: bool,
    show_summary: bool,
    dry_run: bool = False,
    config: ProcessingConfig = DEFAULT_CONFIG,
    log_level: int = logging.WARNING,
) -> int:
    """
    Batch-process images in a folder sequentially.

    Output policy:
    - If emit_jsonl is True: write one JSON report per input file to stdout.
    - Always write errors to stderr via logger.
    - If show_summary is True: write a compact summary line to stderr.

    Returns an exit code (0 ok, 1 partial failures/oversize, 2 fatal input issues).
    """
    input_path = Path(input_folder)
    if not input_path.exists() or not input_path.is_dir():
        raise NotADirectoryError(f"Input folder not found or not a directory: {input_folder}")

    if not dry_run:
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Load face detection model once (reused for all images)
    face_net = load_face_net()

    # Collect all input files
    input_files = [
        entry for entry in sorted(input_path.iterdir())
        if entry.is_file() and entry.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    
    if not input_files:
        logger.warning(f"No image files found in {input_folder}")
        return 0

    total = len(input_files)
    ok = 0
    oversize = 0
    noface = 0
    detector_errors = 0
    open_errors = 0
    downscaled = 0
    has_warnings = 0

    if log_level <= logging.INFO:
        logger.info(f"Processing {total} image(s)...")

    # Sequential processing (reliable and production-ready)
    for entry in input_files:
            out_path = output_path_for_input(output_folder, entry.name)
            report = process_single_image(face_net, str(entry), out_path, config, dry_run)

            if report is None:
                open_errors += 1
                continue

            # Update statistics
            if report.get("face") is None:
                noface += 1
            if report.get("face_error"):
                detector_errors += 1
            if int(report.get("downscale_steps", 0)) > 0:
                downscaled += 1

            status = report.get("status")
            if status == "OK":
                ok += 1
            elif status == "OVERSIZE":
                oversize += 1
            
            warnings = report.get("tass_warnings", [])
            if warnings:
                has_warnings += 1
                if log_level <= logging.INFO:  # Only log warnings if INFO or more verbose
                    for warning in warnings:
                        logger.warning(f"{entry.name}: {warning}")

            if emit_jsonl:
                print(json.dumps(report, ensure_ascii=False))

    # Determine exit code:
    # - 0: all processed files OK (or no matching files)
    # - 1: any OVERSIZE or open errors
    exit_code = 0
    if oversize > 0 or open_errors > 0:
        exit_code = 1

    if show_summary:
        summary_msg = (
            f"Summary: Processed {total} file(s) - "
            f"OK={ok}, OVERSIZE={oversize}, NoFace={noface}, "
            f"DetectorErr={detector_errors}, OpenErr={open_errors}, "
            f"Downscaled={downscaled}, HasWarnings={has_warnings}"
        )
        if log_level <= logging.INFO:
            logger.info(summary_msg)
        else:
            # Always print summary to stderr even if logging is quiet
            print(summary_msg, file=sys.stderr)

    return exit_code


# -----------------------------
# Main entry point
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Center and resize images using face detection for TASS student management system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_folder", help="Folder containing input images.")
    parser.add_argument("output_folder", help="Folder where processed images will be saved.")
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Emit one JSON report per file to stdout (useful for machine processing).",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress the end-of-run summary line.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process images but don't write output files (for testing).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: WARNING).",
    )
    
    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )

    emit_jsonl = bool(args.jsonl)
    show_summary = not bool(args.no_summary)
    dry_run = bool(args.dry_run)

    if dry_run:
        logger.info("DRY RUN MODE - no files will be written")

    try:
        return process_images_in_folder(
            args.input_folder,
            args.output_folder,
            emit_jsonl=emit_jsonl,
            show_summary=show_summary and (not emit_jsonl),
            dry_run=dry_run,
            config=DEFAULT_CONFIG,
            log_level=log_level,
        )
    except FileNotFoundError as e:
        logger.critical(f"Fatal: {e}")
        return 2
    except NotADirectoryError as e:
        logger.critical(f"Fatal: {e}")
        return 2
    except Exception as e:
        logger.critical(f"Fatal: Unexpected error: {e}", exc_info=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
