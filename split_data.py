import shutil
import random
import hashlib
from pathlib import Path

SRC_DIR = Path(r"C:\Users\User\Downloads\Tammi Oi Similar words speakings")
TRAIN_DIR = Path(r"C:\Users\User\Documents\Workspace\Code\python\speech\tammi_oi_2\negative\train\similar_kt")
VALID_DIR = Path(r"C:\Users\User\Documents\Workspace\Code\python\speech\tammi_oi_2\negative\valid\similar_kt")

TRAIN_RATIO = 0.8
SEED = 42


def collect_wav_files(src_root: Path):
    wav_files = []
    if not src_root.exists():
        return wav_files

    for top_dir in sorted(src_root.iterdir()):
        if not top_dir.is_dir():
            continue

        for sub_dir in top_dir.iterdir():
            if not sub_dir.is_dir() or sub_dir.name.lower() not in {"sentence", "word"}:
                continue

            for path in sub_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() == ".wav":
                    wav_files.append(path)

    return sorted(wav_files)


def clear_dir_recursive(dir_path: Path):
    if not dir_path.exists():
        return

    for file_path in dir_path.rglob("*"):
        if file_path.is_file():
            file_path.unlink()

    for folder_path in sorted((p for p in dir_path.rglob("*") if p.is_dir()), reverse=True):
        try:
            folder_path.rmdir()
        except OSError:
            pass


def build_output_name(src_path: Path, src_root: Path, used_names):
    rel = src_path.relative_to(src_root).as_posix()
    digest = hashlib.md5(rel.encode("utf-8")).hexdigest()[:10]
    candidate = f"{src_path.stem}_{digest}{src_path.suffix.lower()}"

    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    idx = 1
    while True:
        fallback = f"{src_path.stem}_{digest}_{idx}{src_path.suffix.lower()}"
        if fallback not in used_names:
            used_names.add(fallback)
            return fallback
        idx += 1


def main():
    wav_files = collect_wav_files(SRC_DIR)
    if not wav_files:
        print(f"Không tìm thấy file WAV trong cây thư mục: {SRC_DIR}")
        return

    random.seed(SEED)
    random.shuffle(wav_files)

    split_idx = int(len(wav_files) * TRAIN_RATIO)
    train_files = wav_files[:split_idx]
    valid_files = wav_files[split_idx:]

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VALID_DIR.mkdir(parents=True, exist_ok=True)

    clear_dir_recursive(TRAIN_DIR)
    clear_dir_recursive(VALID_DIR)
    print("Đã xóa toàn bộ file cũ trong TRAIN_DIR và VALID_DIR.")

    def copy_files(file_list, dest_dir, label):
        copied = 0
        used_names = set()
        for wav_path in file_list:
            out_name = build_output_name(wav_path, SRC_DIR, used_names)
            shutil.copy2(wav_path, dest_dir / out_name)
            copied += 1
        print(f"{label}: {copied} files → {dest_dir}")

    copy_files(train_files, TRAIN_DIR, "Train (80%)")
    copy_files(valid_files, VALID_DIR, "Valid (20%)")
    print(f"\nTổng: {len(wav_files)} WAV files | Train: {len(train_files)} | Valid: {len(valid_files)}")

if __name__ == "__main__":
    main()
