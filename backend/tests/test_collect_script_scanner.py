from pathlib import Path

from app.services.collect_script_scanner import scan_collect_script_file


def test_scan_rm75_compressed_script():
    script = Path("/home/sia/rm75_offline_deb_bundle/test/collect_data_compressed.sh")
    if not script.is_file():
        return
    result = scan_collect_script_file(str(script))
    topics = {t["topic"] for t in result["topics"]}
    assert "/left/joint_states" in topics
    assert "/camera1/camera/color/image_raw/compressed" in topics
    assert result["defaults"]["camera_hz"] == 25.0
    assert result["defaults"]["joint_hz"] == 10.0
