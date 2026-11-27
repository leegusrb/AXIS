import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.model


def test_pose_detects_person_on_sample():
    from ultralytics.utils import ASSETS

    from axis_ai.pose import PoseDetector

    persons = PoseDetector().detect(Image.open(ASSETS / "bus.jpg").convert("RGB"))
    assert len(persons) >= 1
    for p in persons:
        assert len(p.box) == 4 and all(0.0 <= v <= 1.0 for v in p.box)
        assert p.keypoints.shape == (17, 3)
        assert (p.keypoints[:, :2] >= 0).all() and (p.keypoints[:, :2] <= 1).all()
        assert 0.0 <= p.conf <= 1.0


def test_embedder_shape_and_unit_norm():
    from axis_ai.embed import Embedder

    v = Embedder().embed(Image.new("RGB", (300, 400), (120, 80, 40)))
    assert v.shape == (576,) and v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
