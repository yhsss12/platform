from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

CABLE_THREADING_ROOT = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"
sys.path.insert(0, str(CABLE_THREADING_ROOT))

from examples.cable_threading.dp_lab.policy_runtime import _set_inference_mode  # noqa: E402


def test_dp_inference_mode_keeps_vision_batchnorm_in_train_mode():
    model = MagicMock()
    vision = MagicMock()
    vision.training = False
    model.vision = vision

    def _eval():
        model.training = False

    def _vision_train():
        vision.training = True

    model.eval.side_effect = _eval
    vision.train.side_effect = _vision_train

    _set_inference_mode(model)

    model.eval.assert_called_once()
    vision.train.assert_called_once()
    assert model.training is False
    assert vision.training is True


def test_dp_inference_mode_without_vision_only_eval():
    model = MagicMock()
    model.vision = None
    model.eval.side_effect = lambda: setattr(model, "training", False)

    _set_inference_mode(model)

    model.eval.assert_called_once()
    assert model.training is False
