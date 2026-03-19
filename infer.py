import argparse
import csv
import importlib
import math
import wave
from pathlib import Path

import openwakeword


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer an openWakeWord model over one or more test folders."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the custom ONNX/TFLite wake word model.",
    )
    parser.add_argument(
        "--test-dir",
        required=True,
        nargs="+",
        help="One or more folders containing test WAV files.",
    )
    parser.add_argument(
        "--expected-labels",
        nargs="+",
        choices=["positive", "negative", "unknown"],
        help=(
            "Optional expected label for each test dir. Provide one label to apply to all test dirs, "
            "or one label per test dir."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="infer_results.csv",
        help="Path to save inference results as CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Detection threshold applied to max score per file.",
    )
    parser.add_argument(
        "--bucket-size",
        type=float,
        default=0.1,
        help="Bucket size used to group max_score values in the output CSV.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1280,
        help="Audio chunk size passed to predict_clip. Default is 1280 samples (80 ms).",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.0,
        help="Optional VAD threshold. Set 0 to disable VAD.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subfolders for WAV files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip files that fail to infer and keep processing the rest.",
    )
    parser.add_argument(
        "--speex-noise-suppression",
        action="store_true",
        help="Enable Speex noise suppression before feature extraction.",
    )
    return parser.parse_args()


def find_audio_files(test_dir: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    return sorted(
        path for path in test_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def get_wav_duration_seconds(wav_path: Path):
    try:
        with wave.open(str(wav_path), "rb") as handle:
            frames = handle.getnframes()
            sample_rate = handle.getframerate()
        return frames / float(sample_rate) if sample_rate else 0.0
    except Exception:
        # Fallback 1: Try scipy (no external DLL dependencies)
        try:
            scipy_io = importlib.import_module("scipy.io")
            sample_rate, _ = scipy_io.wavfile.read(str(wav_path))
            # scipy doesn't return duration directly, so estimate from file size
            # Read file size and make rough estimate
            file_size = wav_path.stat().st_size
            # WAV header is typically ~44 bytes, rest is audio data
            # Rough estimate: 2 bytes per sample at 16-bit
            estimated_duration = (file_size - 100) / (2 * sample_rate) if sample_rate else 1.0
            return max(estimated_duration, 0.1)
        except Exception:
            # Fallback 2: Try soundfile (does not depend on torchcodec)
            try:
                soundfile = importlib.import_module("soundfile")
                data, sample_rate = soundfile.read(str(wav_path), dtype='float32')
                return len(data) / float(sample_rate) if sample_rate else 1.0
            except Exception:
                # Fallback 3: Parse WAV header manually
                try:
                    with open(wav_path, 'rb') as f:
                        # Read WAV header manually to extract duration
                        f.read(12)  # Skip RIFF header
                        while True:
                            chunk_id = f.read(4)
                            if not chunk_id:
                                break
                            chunk_size = int.from_bytes(f.read(4), 'little')
                            if chunk_id == b'fmt ':
                                fmt_data = f.read(chunk_size)
                                if len(fmt_data) >= 14:
                                    channels = int.from_bytes(fmt_data[2:4], 'little')
                                    sample_rate = int.from_bytes(fmt_data[4:8], 'little')
                                    break
                            else:
                                f.seek(f.tell() + chunk_size)
                        f.seek(0)
                        # Find data chunk
                        f.read(12)
                        while True:
                            chunk_id = f.read(4)
                            chunk_size = int.from_bytes(f.read(4), 'little')
                            if chunk_id == b'data':
                                if sample_rate:
                                    num_frames = chunk_size // (channels * 2)  # Assume 16-bit
                                    return num_frames / float(sample_rate)
                            f.seek(f.tell() + chunk_size)
                except Exception:
                    pass
                # Final fallback: return 1.0 second
                return 1.0


def normalize_expected_labels(test_dirs, expected_labels):
    if not expected_labels:
        return ["unknown"] * len(test_dirs)
    if len(expected_labels) == 1:
        return expected_labels * len(test_dirs)
    if len(expected_labels) != len(test_dirs):
        raise ValueError(
            "--expected-labels must contain either one value or one value per --test-dir"
        )
    return expected_labels


def score_to_bucket(score: float, bucket_size: float):
    if bucket_size <= 0:
        raise ValueError("--bucket-size must be greater than 0")

    bucket_index = min(int(math.floor(score / bucket_size)), int(math.floor(1.0 / bucket_size)))
    bucket_start = bucket_index * bucket_size
    bucket_end = min(bucket_start + bucket_size, 1.0)
    return f"{bucket_start:.2f}-{bucket_end:.2f}"


def iter_test_files(test_dirs, expected_labels, recursive):
    for test_dir, expected_label in zip(test_dirs, expected_labels):
        for wav_path in find_audio_files(test_dir, recursive):
            yield test_dir, expected_label, wav_path


def main():
    args = parse_args()

    model_path = Path(args.model_path).resolve()
    test_dirs = [Path(test_dir).resolve() for test_dir in args.test_dir]
    output_csv = Path(args.output_csv).resolve()
    expected_labels = normalize_expected_labels(test_dirs, args.expected_labels)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    for test_dir in test_dirs:
        if not test_dir.exists() or not test_dir.is_dir():
            raise NotADirectoryError(f"Test directory not found: {test_dir}")

    model_name = model_path.stem
    inference_framework = "onnx" if model_path.suffix.lower() == ".onnx" else "tflite"

    if model_path.suffix.lower() not in {".onnx", ".tflite"}:
        raise ValueError("--model-path must point to an .onnx or .tflite model file")

    file_jobs = list(iter_test_files(test_dirs, expected_labels, args.recursive))
    if not file_jobs:
        searched_dirs = ", ".join(str(test_dir) for test_dir in test_dirs)
        raise FileNotFoundError(f"No WAV files found in: {searched_dirs}")

    oww = openwakeword.Model(
        wakeword_models=[str(model_path)],
        inference_framework=inference_framework,
        enable_speex_noise_suppression=args.speex_noise_suppression,
        vad_threshold=args.vad_threshold,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    detected_count = 0
    failed_count = 0
    success_count = 0
    total_duration_seconds = 0.0
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "root_dir",
                "expected_label",
                "relative_path",
                "file",
                "max_score",
                "score_bucket",
                "detected",
                "threshold",
                "n_frames",
                "duration_seconds",
                "error",
            ],
        )
        writer.writeheader()

        for index, (root_dir, expected_label, wav_path) in enumerate(file_jobs, start=1):
            relative_path = wav_path.relative_to(root_dir)
            duration_seconds = 0.0
            error_message = ""
            max_score = 0.0
            score_bucket = score_to_bucket(0.0, args.bucket_size)
            detected = False
            n_frames = 0

            try:
                duration_seconds = get_wav_duration_seconds(wav_path)
                total_duration_seconds += duration_seconds

                oww.reset()
                predictions = oww.predict_clip(str(wav_path), chunk_size=args.chunk_size)
                n_frames = len(predictions)
                max_score = max((frame[model_name] for frame in predictions), default=0.0)
                score_bucket = score_to_bucket(max_score, args.bucket_size)
                detected = max_score >= args.threshold
                success_count += 1
                if detected:
                    detected_count += 1
            except Exception as exc:
                failed_count += 1
                error_message = str(exc)
                if not args.continue_on_error:
                    raise

            writer.writerow(
                {
                    "root_dir": str(root_dir),
                    "expected_label": expected_label,
                    "relative_path": str(relative_path),
                    "file": str(wav_path),
                    "max_score": float(max_score),
                    "score_bucket": score_bucket,
                    "detected": int(detected),
                    "threshold": args.threshold,
                    "n_frames": n_frames,
                    "duration_seconds": duration_seconds,
                    "error": error_message,
                }
            )

            status = f"max_score={max_score:.6f}, detected={detected}"
            if error_message:
                status = f"error={error_message}"
            print(f"[{index}/{len(file_jobs)}] {relative_path}: {status}")

    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Files processed: {len(file_jobs)}")
    print(f"Successful: {success_count}")
    print(f"Detected: {detected_count}")
    print(f"Failed: {failed_count}")
    print(f"Threshold: {args.threshold}")
    print(f"Bucket size: {args.bucket_size}")
    print(f"Audio hours: {total_duration_seconds / 3600:.4f}")
    if total_duration_seconds > 0:
        print(f"Detections per hour: {detected_count / (total_duration_seconds / 3600):.4f}")
    print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    main()