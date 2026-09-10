"""Prepare and validate resources used by the CRLD-Proto experiments.

The script is intentionally idempotent. It never treats a placeholder file or
an incomplete download as a dataset/checkpoint. ImageNet itself is not
downloaded because access is governed by the dataset license; only the public
PyTorch teacher weights are cached here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "download_ckpts"
MANIFEST_PATH = ROOT / "experiments" / "resources_manifest.json"
EXPERIMENT_IDENTITY = {
    "method": "CRLD-Proto",
    "baseline": "CRLD",
    "relationship": (
        "CRLD-Proto preserves the complete CRLD objective and adds "
        "multi-center prototype correction to the teacher strong-view target."
    ),
}

TINY_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_EXPECTED_BYTES = 248_100_043
IMAGE_NET_TEACHERS = {
    "resnet34": (
        "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
        "ResNet34_vanilla/ckpt_epoch_90.pth",
    ),
    "resnet50": (
        "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        "ResNet50_vanilla/ckpt_epoch_90.pth",
    ),
}

CIFAR_TEACHERS = {
    "resnet56": "resnet56_vanilla/ckpt_epoch_240.pth",
    "resnet110": "resnet110_vanilla/ckpt_epoch_240.pth",
    "resnet32x4": "resnet32x4_vanilla/ckpt_epoch_240.pth",
    "wrn_40_2": "wrn_40_2_vanilla/ckpt_epoch_240.pth",
    "vgg13": "vgg13_vanilla/ckpt_epoch_240.pth",
    "ResNet50": "ResNet50_vanilla/ckpt_epoch_240.pth",
}


def report(message: str, ok: bool | None = None) -> None:
    if ok is True:
        prefix = "[OK]"
    elif ok is False:
        prefix = "[MISSING]"
    else:
        prefix = "[INFO]"
    print(f"{prefix} {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_resumable(
    url: str,
    destination: Path,
    expected_size: int | None = None,
    validator: Callable[[Path], bool] | None = None,
) -> None:
    """Download a file with resume support and an atomic final rename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    current_size = partial.stat().st_size if partial.exists() else 0

    # A partial file from a different mirror can be larger than the official
    # archive. Never issue an invalid Range request in that case.
    if expected_size is not None and current_size > expected_size:
        partial.unlink()
        current_size = 0

    # An exact-length partial file may still be corrupt or encrypted. Validate
    # it before treating it as a completed download.
    if current_size and expected_size is not None and current_size == expected_size:
        if validator is None or validator(partial):
            os.replace(str(partial), str(destination))
            return
        partial.unlink()
        current_size = 0

    request = urllib.request.Request(url)
    if current_size:
        request.add_header("Range", f"bytes={current_size}-")

    report(f"Downloading {url} -> {destination}")
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", response.getcode())
        append = current_size > 0 and status == 206
        if not append:
            current_size = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)

    actual_size = partial.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise RuntimeError(
            f"Incomplete download: {partial} has {actual_size} bytes; "
            f"expected {expected_size}. Run the command again to resume."
        )
    if validator is not None and not validator(partial):
        raise RuntimeError(f"Downloaded file failed validation: {partial}")
    os.replace(str(partial), str(destination))


def tiny_archive_is_usable(path: Path) -> bool:
    """Check the ZIP structure before extraction, including encryption flags."""
    if not path.is_file() or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if any(info.flag_bits & 1 for info in infos):
                return False
            names = {info.filename for info in infos}
            return any(name.endswith("/wnids.txt") for name in names) and any(
                name.endswith("/train/") for name in names
            )
    except (OSError, zipfile.BadZipFile):
        return False


def safe_extract(zip_path: Path, target: Path) -> None:
    target = target.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        encrypted = [info.filename for info in archive.infolist() if info.flag_bits & 1]
        if encrypted:
            raise RuntimeError(
                "Tiny-ImageNet archive contains encrypted members and cannot be used: "
                f"{encrypted[0]}"
            )
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"Corrupt Tiny-ImageNet archive member: {bad_member}")
        for member in archive.infolist():
            member_path = (target / member.filename).resolve()
            if target not in member_path.parents and member_path != target:
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
        archive.extractall(target)


def tiny_root_is_ready(root: Path) -> bool:
    train = root / "train"
    val = root / "val"
    if not (train.is_dir() and val.is_dir() and (root / "wnids.txt").is_file()):
        return False
    train_classes = [path for path in train.iterdir() if path.is_dir()]
    val_classes = [path for path in val.iterdir() if path.is_dir()]
    if len(train_classes) != 200 or len(val_classes) != 200:
        return False
    train_images = list(train.glob("*/images/*"))
    val_images = list(val.glob("*/images/*"))
    return len(train_images) == 100_000 and len(val_images) == 10_000


def organize_tiny_validation(root: Path) -> None:
    """Convert the official flat validation directory to ImageFolder format."""
    val = root / "val"
    annotations = val / "val_annotations.txt"
    if not annotations.is_file():
        return

    rows = []
    with annotations.open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.strip().split()
            if len(fields) >= 2:
                rows.append((fields[0], fields[1]))

    for image_name, class_name in rows:
        source = val / "images" / image_name
        if not source.exists():
            source = val / image_name
        if not source.exists():
            raise RuntimeError(f"Validation image listed but not found: {image_name}")
        destination_dir = val / class_name / "images"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / image_name
        if not destination.exists():
            shutil.move(str(source), str(destination))

    flat_images = val / "images"
    if flat_images.is_dir() and not any(flat_images.iterdir()):
        flat_images.rmdir()
    annotations.unlink(missing_ok=True)


def prepare_tiny(url: str, expected_size: int | None = None) -> dict:
    root = DATA_DIR / "tiny-imagenet-200"
    archive = DATA_DIR / "tiny-imagenet-200.zip"
    partial = archive.with_suffix(archive.suffix + ".part")
    if expected_size is None and "cs231n.stanford.edu" in url:
        expected_size = TINY_EXPECTED_BYTES
    if not tiny_root_is_ready(root):
        archive_ready = tiny_archive_is_usable(archive)
        if not archive_ready or (
            expected_size is not None and archive.stat().st_size != expected_size
        ):
            download_resumable(
                url,
                archive,
                expected_size,
                validator=tiny_archive_is_usable,
            )
        staging = DATA_DIR / ".tiny-imagenet-200-extract"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        safe_extract(archive, staging)
        extracted = staging / "tiny-imagenet-200"
        if not extracted.is_dir():
            raise RuntimeError("Tiny-ImageNet archive has an unexpected top-level directory")
        organize_tiny_validation(extracted)
        if root.exists() and root.is_file():
            root.unlink()
        if root.exists():
            shutil.rmtree(root)
        extracted.replace(root)
        shutil.rmtree(staging, ignore_errors=True)
    elif partial.exists():
        # Do not leave a stale partial download beside a validated dataset.
        partial.unlink()

    ready = tiny_root_is_ready(root)
    report(f"Tiny-ImageNet: {root}", ready)
    return {
        "status": "ready" if ready else "missing",
        "path": str(root),
        "classes": 200 if ready else 0,
        "train_images": 100_000 if ready else 0,
        "val_images": 10_000 if ready else 0,
    }


def check_cifar() -> dict:
    root = DATA_DIR / "cifar-100-python"
    archive = DATA_DIR / "cifar-100-python.tar.gz"
    data_ready = root.is_dir() and (root / "train").is_file() and (root / "test").is_file()
    report(f"CIFAR-100 data: {root}", data_ready)

    teacher_status = {}
    teacher_root = CKPT_DIR / "cifar_teachers"
    for name, relative_path in CIFAR_TEACHERS.items():
        path = teacher_root / relative_path
        ready = path.is_file() and path.stat().st_size > 1_000_000
        teacher_status[name] = {"status": "ready" if ready else "missing", "path": str(path)}
        report(f"CIFAR teacher {name}: {path}", ready)
    return {
        "status": "ready" if data_ready and all(item["status"] == "ready" for item in teacher_status.values()) else "missing",
        "data_path": str(root),
        "archive": str(archive) if archive.is_file() else None,
        "teachers": teacher_status,
    }


def check_imagenet_data() -> dict:
    root = DATA_DIR / "imagenet"
    train_ready = (root / "train").is_dir()
    val_ready = (root / "val").is_dir()
    ready = train_ready and val_ready
    report(
        "ImageNet data: expected ImageFolder directories "
        f"{root / 'train'} and {root / 'val'}",
        ready,
    )
    return {
        "status": "ready" if ready else "external_required",
        "path": str(root),
        "train": str(root / "train"),
        "val": str(root / "val"),
    }


def prepare_imagenet_teachers() -> dict:
    root = CKPT_DIR / "imagenet_teachers"
    result = {}
    for name, (url, filename) in IMAGE_NET_TEACHERS.items():
        path = root / filename
        if not path.is_file() or path.stat().st_size < 10_000_000:
            download_resumable(url, path)
        digest = sha256(path)
        result[name] = {"status": "ready", "path": str(path), "sha256": digest, "url": url}
        report(f"ImageNet teacher {name}: {path}", True)
    return result


def check_tiny_teacher() -> dict:
    path = CKPT_DIR / "tinyimagenet_teachers" / "resnet32x4_vanilla" / "ckpt_epoch_240.pth"
    ready = path.is_file() and path.stat().st_size > 10_000_000
    report(
        f"Tiny-ImageNet ResNet32x4 teacher: {path} "
        "(the CRLD authors distribute this privately, not in a public release)",
        ready,
    )
    return {
        "status": "ready" if ready else "external_required",
        "path": str(path),
        "source": "CRLD author-provided checkpoint or local teacher training",
    }


def write_manifest(resources: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(resources, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report(f"Resource manifest: {MANIFEST_PATH}", True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-public",
        action="store_true",
        help="Download and prepare Tiny-ImageNet and public ImageNet teacher weights.",
    )
    parser.add_argument(
        "--tiny-url",
        default=TINY_URL,
        help="Override the Tiny-ImageNet archive URL when a local mirror is preferred.",
    )
    parser.add_argument(
        "--tiny-expected-bytes",
        type=int,
        default=0,
        help="Expected size of a mirror archive. Zero validates the ZIP directly.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return failure until ImageNet data and the Tiny-ImageNet teacher are also ready.",
    )
    args = parser.parse_args()

    resources = {
        "experiment": EXPERIMENT_IDENTITY,
        "root": str(ROOT),
        "cifar100": check_cifar(),
    }
    if args.prepare_public:
        resources["tinyimagenet200"] = prepare_tiny(
            args.tiny_url,
            args.tiny_expected_bytes or None,
        )
        resources["imagenet_teachers"] = prepare_imagenet_teachers()
    else:
        resources["tinyimagenet200"] = {
            "status": "ready" if tiny_root_is_ready(DATA_DIR / "tiny-imagenet-200") else "missing",
            "path": str(DATA_DIR / "tiny-imagenet-200"),
        }
        resources["imagenet_teachers"] = {
            "status": "check-only",
            "paths": [str(CKPT_DIR / "imagenet_teachers" / filename) for _, filename in IMAGE_NET_TEACHERS.values()],
        }
        report("Use --prepare-public to download public resources.")
    resources["imagenet"] = check_imagenet_data()
    resources["tiny_teacher"] = check_tiny_teacher()
    write_manifest(resources)

    required = [resources["cifar100"]["status"] == "ready"]
    if args.prepare_public:
        required.extend([
            resources["tinyimagenet200"]["status"] == "ready",
            all(item["status"] == "ready" for item in resources["imagenet_teachers"].values()),
        ])
    if args.strict:
        required.extend([
            resources["imagenet"]["status"] == "ready",
            resources["tiny_teacher"]["status"] == "ready",
        ])
    if not all(required):
        report(
            "Some resources still require user action. See experiments/resources_manifest.json.",
            False,
        )
        return 2
    report("Public resource preparation completed.", True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
