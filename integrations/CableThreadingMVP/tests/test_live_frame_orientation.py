import numpy as np

from examples.cable_threading.utils import normalize_live_rgb_frame, stabilize_live_display_frame


def test_normalize_live_rgb_frame_flips_opengl_origin_once():
    raw = np.zeros((4, 6, 3), dtype=np.uint8)
    raw[0, :, 0] = 255
    display = normalize_live_rgb_frame(raw)
    assert display is not None
    assert display[-1, 0, 0] == 255
    assert display[0, 0, 0] == 0


def test_stabilize_live_display_frame_keeps_orientation_when_buffers_alternate():
    live_config: dict = {}
    base = np.random.default_rng(0).integers(0, 255, size=(32, 48, 3), dtype=np.uint8)
    first = stabilize_live_display_frame(base.copy(), live_config)
    second = stabilize_live_display_frame(base[::-1].copy(), live_config)
    third = stabilize_live_display_frame(base.copy(), live_config)

    assert np.array_equal(first, second)
    assert np.array_equal(first, third)
