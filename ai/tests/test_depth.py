import base64
import io

import numpy as np
import pytest
from PIL import Image

from axis_ai.depth import DepthFeatures, depth_features, preview_png


def test_depth_features_inside_outside():
    d = np.zeros((100, 100), np.float32)
    d[20:60, 30:70] = 0.9
    f = depth_features(d, (0.3, 0.2, 0.4, 0.4))
    assert f.subject == pytest.approx(0.9) and f.background == pytest.approx(0.0)
    assert f.diff == pytest.approx(0.9)
    assert f.vector().shape == (3,)


def test_depth_features_full_frame_box():
    d = np.full((10, 10), 0.4, np.float32)
    f = depth_features(d, (0.0, 0.0, 1.0, 1.0))
    assert f == DepthFeatures(pytest.approx(0.4), pytest.approx(0.4), pytest.approx(0.0))


def test_preview_png_width():
    png = base64.b64decode(preview_png(np.random.rand(90, 120).astype(np.float32), width=64))
    img = Image.open(io.BytesIO(png))
    assert img.size == (64, 48) and img.mode == "L"


@pytest.mark.model
def test_depth_estimator_real():
    from axis_ai.depth import DepthEstimator
    d = DepthEstimator(size=256).estimate(Image.new("RGB", (640, 480), (100, 120, 140)))
    assert min(d.shape) == 256 and d.dtype == np.float32
    assert 0.0 <= d.min() and d.max() <= 1.0
