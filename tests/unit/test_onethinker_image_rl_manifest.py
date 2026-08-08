from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/lkl_8gpu/easyr1/build_onethinker_image_rl_manifest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("onethinker_image_manifest", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


manifest = load_module()


class OneThinkerImageManifestTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path]:
        media_root = root / "media"
        image_path = media_root / "images" / "sample.png"
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(b"image")
        source_file = root / "source.json"
        source_file.write_text(
            json.dumps(
                [
                    {"problem_id": "image-1", "data_type": "image", "images": ["./images/sample.png"]},
                    {"problem_id": "video-1", "data_type": "video", "videos": ["./videos/sample.mp4"]},
                ]
            ),
            encoding="utf-8",
        )
        return source_file, media_root

    def test_filters_only_image_records_and_preserves_record_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file, media_root = self.make_fixture(root)
            source_records, image_records = manifest.load_image_records(source_file, media_root.resolve(), 1)

        self.assertEqual(len(source_records), 2)
        self.assertEqual(image_records, [source_records[0]])
        self.assertEqual(image_records[0]["images"], ["./images/sample.png"])

    def test_missing_media_blocks_manifest_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file, media_root = self.make_fixture(root)
            source_file.write_text(
                json.dumps([{"data_type": "image", "images": ["images/missing.png"]}]), encoding="utf-8"
            )

            with self.assertRaises(FileNotFoundError):
                manifest.load_image_records(source_file, media_root.resolve(), 1)

    def test_path_escape_blocks_manifest_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file, media_root = self.make_fixture(root)
            source_file.write_text(
                json.dumps([{"data_type": "image", "images": ["../outside.png"]}]), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "escapes media root"):
                manifest.load_image_records(source_file, media_root.resolve(), 1)

    def test_expected_count_blocks_partial_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file, media_root = self.make_fixture(root)

            with self.assertRaisesRegex(ValueError, "expected 2 image records, found 1"):
                manifest.load_image_records(source_file, media_root.resolve(), 2)

    def test_atomic_write_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_file = Path(directory) / "manifest.json"
            manifest.write_json_atomically(output_file, [{"problem_id": "first"}], overwrite=False)
            with self.assertRaises(FileExistsError):
                manifest.write_json_atomically(output_file, [{"problem_id": "second"}], overwrite=False)
            manifest.write_json_atomically(output_file, [{"problem_id": "second"}], overwrite=True)

            self.assertEqual(json.loads(output_file.read_text(encoding="utf-8")), [{"problem_id": "second"}])
