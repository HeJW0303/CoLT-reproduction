#!/usr/bin/env python3

import hashlib
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


DEEPSPEED_VERSION = "0.16.9"
SDIST_URL = (
    "https://files.pythonhosted.org/packages/4f/5c/"
    "7542904cddfaa50a9a7ae6770349d468773359f5af1718865452cea8729d/"
    "deepspeed-0.16.9.tar.gz"
)
SDIST_SHA256 = "88dc08986cd321047c37c2d0f1edd6faac498a1b52eb486d559164362c6b9011"

ORIGINAL = "if torch_available and get_accelerator().device_name() == 'cuda':\n    cupy = None\n"
PATCHED = (
    "if (torch_available and get_accelerator().device_name() == 'cuda'\n"
    "        and os.environ.get('DS_BUILD_OPS', '0') == '1'):\n"
    "    cupy = None\n"
)
ORIGINAL_OP_CHECK = "for op_name, builder in ALL_OPS.items():\n    op_compatible = builder.is_compatible()\n"
PATCHED_OP_CHECK = (
    "for op_name, builder in ALL_OPS.items():\n"
    "    op_compatible = builder.is_compatible() if op_enabled(op_name) else False\n"
)


def download(url: str, destination: Path) -> None:
    last_error = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            return
        except Exception as error:
            last_error = error
            destination.unlink(missing_ok=True)
            print(f"DeepSpeed download attempt {attempt}/3 failed: {error}", file=sys.stderr)
    raise RuntimeError("Unable to download the pinned DeepSpeed source distribution") from last_error


def safe_extract(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError(f"Unsafe path in DeepSpeed archive: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Links are not allowed in DeepSpeed archive: {member.name}")
        handle.extractall(destination)
    source_root = destination / f"deepspeed-{DEEPSPEED_VERSION}"
    if not (source_root / "setup.py").is_file():
        raise RuntimeError(f"DeepSpeed setup.py is missing under {source_root}")
    return source_root


def patch_setup(source_root: Path) -> None:
    setup_path = source_root / "setup.py"
    source = setup_path.read_text(encoding="utf-8")
    if source.count(ORIGINAL) != 1 or source.count(ORIGINAL_OP_CHECK) != 1:
        raise RuntimeError("Pinned DeepSpeed setup.py no longer matches the expected source")
    source = source.replace(ORIGINAL, PATCHED)
    source = source.replace(ORIGINAL_OP_CHECK, PATCHED_OP_CHECK)
    setup_path.write_text(source, encoding="utf-8")


def main() -> None:
    try:
        if version("deepspeed") == DEEPSPEED_VERSION:
            print(f"DeepSpeed {DEEPSPEED_VERSION} is already installed")
            return
    except PackageNotFoundError:
        pass

    temp_parent = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deepspeed-install-", dir=temp_parent) as temp_dir:
        temp_root = Path(temp_dir)
        archive = temp_root / f"deepspeed-{DEEPSPEED_VERSION}.tar.gz"
        download(SDIST_URL, archive)
        actual_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual_hash != SDIST_SHA256:
            raise RuntimeError(f"DeepSpeed sdist SHA256 mismatch: {actual_hash}")

        source_root = safe_extract(archive, temp_root)
        patch_setup(source_root)
        env = os.environ.copy()
        env["DS_BUILD_OPS"] = "0"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", str(source_root), "--no-build-isolation"],
            check=True,
            env=env,
        )

    if version("deepspeed") != DEEPSPEED_VERSION:
        raise RuntimeError("DeepSpeed version verification failed after installation")
    print(f"DeepSpeed {DEEPSPEED_VERSION} installed without precompiled CUDA ops")


if __name__ == "__main__":
    main()
