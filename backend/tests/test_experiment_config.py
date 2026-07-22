import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.experiment import ExperimentMethodUpdateRequest
from app.services.experiment_config import ExperimentConfigService


def test_load_missing_config_writes_default(tmp_path):
    path = tmp_path / "experiment_method.yaml"
    svc = ExperimentConfigService(path)

    out = svc.load()

    assert out.experiment_method.name == "proposed"
    assert out.experiment_method.method_code == "P"
    assert path.exists()


def test_save_switches_method_profile(tmp_path):
    path = tmp_path / "experiment_method.yaml"
    svc = ExperimentConfigService(path)

    svc.save(ExperimentMethodUpdateRequest(name="baseline_b3"))
    out = svc.load()

    assert out.experiment_method.name == "baseline_b3"
    assert out.experiment_method.method_code == "B3"
    assert out.experiment_method.preview_mode_lock == "auto"
    assert out.experiment_method.browser_recovery_enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
