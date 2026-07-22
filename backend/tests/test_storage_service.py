"""StorageService URI 解析与规范化测试。"""

from pathlib import Path

from app.services.storage.storage_service import StorageService


def test_parse_minio_uri():
    parsed = StorageService.parse_uri("minio://eai-checkpoints/checkpoints/train_1/model.ckpt")
    assert parsed.scheme == "minio"
    assert parsed.bucket == "eai-checkpoints"
    assert parsed.object_key == "checkpoints/train_1/model.ckpt"


def test_parse_file_uri():
    parsed = StorageService.parse_uri("file:///tmp/demo.ckpt")
    assert parsed.scheme == "file"
    assert parsed.local_path == Path("/tmp/demo.ckpt")


def test_normalize_absolute_path_to_file_uri(tmp_path):
    file_path = tmp_path / "artifact.bin"
    file_path.write_bytes(b"data")
    uri = StorageService.normalize_uri(str(file_path))
    assert uri.startswith("file://")
    assert StorageService.local_path_from_uri(uri) == file_path.resolve()


def test_parse_uri_dict():
    parsed = StorageService.parse_uri_dict("minio://eai-checkpoints/checkpoints/train_1/model.ckpt")
    assert parsed["scheme"] == "minio"
    assert parsed["bucket"] == "eai-checkpoints"
    assert parsed["key"] == "checkpoints/train_1/model.ckpt"


def test_parse_file_uri_dict():
    parsed = StorageService.parse_uri_dict("file:///tmp/demo.ckpt")
    assert parsed["scheme"] == "file"
    assert parsed["bucket"] is None
    assert parsed["key"] == "/tmp/demo.ckpt"
