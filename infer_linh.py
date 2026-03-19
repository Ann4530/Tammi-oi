import argparse
import importlib
import math
import os
import wave
import shutil
from pathlib import Path

import openwakeword


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer an openWakeWord model and copy files into score buckets."
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
        "--output-dir",
        default="./output_chunks",
        help="Directory where the 10 chunk folders will be created.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Detection threshold applied to max score per file (used for console stats).",
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
        # Fallback 1: Try scipy
        try:
            scipy_io = importlib.import_module("scipy.io")
            sample_rate, _ = scipy_io.wavfile.read(str(wav_path))
            file_size = wav_path.stat().st_size
            estimated_duration = (file_size - 100) / (2 * sample_rate) if sample_rate else 1.0
            return max(estimated_duration, 0.1)
        except Exception:
            # Fallback 2: Try soundfile
            try:
                soundfile = importlib.import_module("soundfile")
                data, sample_rate = soundfile.read(str(wav_path), dtype='float32')
                return len(data) / float(sample_rate) if sample_rate else 1.0
            except Exception:
                # Fallback 3: Parse WAV header manually
                try:
                    with open(wav_path, 'rb') as f:
                        f.read(12)  # Skip RIFF header
                        while True:
                            chunk_id = f.read(4)
                            if not chunk_id: break
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
                        f.read(12)
                        while True:
                            chunk_id = f.read(4)
                            chunk_size = int.from_bytes(f.read(4), 'little')
                            if chunk_id == b'data':
                                if sample_rate:
                                    num_frames = chunk_size // (channels * 2)
                                    return num_frames / float(sample_rate)
                            f.seek(f.tell() + chunk_size)
                except Exception:
                    pass
                return 1.0


def score_to_folder_name(score: float):
    """Phân loại điểm phù hợp với logic tạo thư mục trong hàm main."""
    # Đảm bảo điểm nằm trong khoảng 0.0 -> 1.0
    score = max(0.0, min(1.0, score))

    if score < 0.1:
        # Từ 0.0 đến < 0.1: Phân loại thành 20 buckets (step = 0.005)
        bucket_idx = int(score / 0.001)
        bucket_idx = min(bucket_idx, 19)  # Đảm bảo tối đa là bucket thứ 19
        
        start = bucket_idx * 0.001
        end = start + 0.001
        return f"{start:.3f}-{end:.3f}"
    else:
        # Từ 0.1 đến 1.0: Phân loại thành 9 buckets (step = 0.1)
        bucket_idx = int(score * 10)
        bucket_idx = min(bucket_idx, 9)  # Đảm bảo tối đa là bucket thứ 9 (0.90-1.00)
        
        start = bucket_idx * 0.1
        end = start + 0.1
        return f"{start:.2f}-{end:.2f}"


def iter_test_files(test_dirs, recursive):
    for test_dir in test_dirs:
        for wav_path in find_audio_files(test_dir, recursive):
            yield test_dir, wav_path


def main():
    args = parse_args()

    model_path = Path(args.model_path).resolve()
    test_dirs = [Path(test_dir).resolve() for test_dir in args.test_dir]
    output_dir =  (Path.cwd() / args.output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    for test_dir in test_dirs:
        if not test_dir.exists() or not test_dir.is_dir():
            raise NotADirectoryError(f"Test directory not found: {test_dir}")

    model_name = model_path.stem
    inference_framework = "onnx" if model_path.suffix.lower() == ".onnx" else "tflite"

    if model_path.suffix.lower() not in {".onnx", ".tflite"}:
        raise ValueError("--model-path must point to an .onnx or .tflite model file")

    file_jobs = list(iter_test_files(test_dirs, args.recursive))
    if not file_jobs:
        searched_dirs = ", ".join(str(test_dir) for test_dir in test_dirs)
        raise FileNotFoundError(f"No WAV files found in: {searched_dirs}")

    oww = openwakeword.Model(
        wakeword_models=[str(model_path)],
        inference_framework=inference_framework,
        enable_speex_noise_suppression=args.speex_noise_suppression,
        vad_threshold=args.vad_threshold,
    )

    # Tạo output_dir và 10 thư mục chunk bên trong
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_folders = {}
    for i in range(10):
        if i == 0:
            for j in range(100):
                start = j * 0.001
                end = start + 0.001
                folder_name = f"{start:.3f}-{end:.3f}"
                folder_path = output_dir / folder_name
                folder_path.mkdir(exist_ok=True)
                chunk_folders[folder_name] = folder_path
        else:
            start = i * 0.1
            end = start + 0.1
            folder_name = f"{start:.2f}-{end:.2f}"
            folder_path = output_dir / folder_name
            folder_path.mkdir(exist_ok=True)
            chunk_folders[folder_name] = folder_path

    detected_count = 0
    failed_count = 0
    success_count = 0
    total_duration_seconds = 0.0

    print(f"Bắt đầu phân loại file vào {output_dir}...")
    
    for index, (root_dir, wav_path) in enumerate(file_jobs, start=1):
        relative_path = wav_path.relative_to(root_dir)
        error_message = ""
        max_score = 0.0
        detected = False

        try:
            duration_seconds = get_wav_duration_seconds(wav_path)
            total_duration_seconds += duration_seconds

            oww.reset()
            predictions = oww.predict_clip(str(wav_path), chunk_size=args.chunk_size)
            max_score = max((frame[model_name] for frame in predictions), default=0.0)
            detected = max_score >= args.threshold
            
            # Chọn thư mục đích để copy
            target_folder_name = score_to_folder_name(max_score)
            target_folder = chunk_folders[target_folder_name]

            # Xử lý tránh ghi đè file nếu bị trùng tên
            dest_file = target_folder / wav_path.name
            counter = 1
            while dest_file.exists():
                dest_file = target_folder / f"{wav_path.stem}_{counter}{wav_path.suffix}"
                counter += 1

            # Copy file vào chunk folder
            shutil.copy2(wav_path, dest_file)
            
            success_count += 1
            if detected:
                detected_count += 1
        except Exception as exc:
            failed_count += 1
            error_message = str(exc)
            if not args.continue_on_error:
                raise

        status = f"max_score={max_score:.6f} -> copied to [{target_folder_name}]"
        if error_message:
            status = f"error={error_message}"
        print(f"[{index}/{len(file_jobs)}] {relative_path}: {status}")

    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Files processed: {len(file_jobs)}")
    print(f"Successful: {success_count}")
    print(f"Detected (>= {args.threshold}): {detected_count}")
    print(f"Failed: {failed_count}")
    print(f"Audio hours: {total_duration_seconds / 3600:.4f}")
    if total_duration_seconds > 0:
        print(f"Detections per hour: {detected_count / (total_duration_seconds / 3600):.4f}")
    print(f"Đã phân loại file vào thư mục: {output_dir}")


if __name__ == "__main__":
    main()