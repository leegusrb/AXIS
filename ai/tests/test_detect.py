import numpy as np
import pytest
from axis_ai.artists import RULE_ARTISTS
from axis_ai.detect import FOOD_CLASSES, boxes_to_food
from axis_ai.features import compute_features
from axis_ai.guide import decide, message_for


def test_boxes_to_food_keeps_food_only():
    xyxy = np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.9, 0.9], [0.0, 0.0, 1.0, 1.0]])
    out = boxes_to_food(xyxy, np.array([0.9, 0.8, 0.95]), ["pizza", "person", "dining table"])
    assert [p.label for p in out] == ["pizza"]
    assert out[0].keypoints.shape == (17, 3) and out[0].keypoints[:, 2].max() == 0


def test_food_features_and_message():
    p = boxes_to_food(np.array([[0.3, 0.3, 0.7, 0.7]]), np.array([0.9]), ["cake"])[0]
    f = compute_features(p)
    assert abs(f.cx - 0.5) < 1e-9 and abs(f.area - 0.16) < 1e-9
    assert decide(None, RULE_ARTISTS["food_center"], noun="음식").message == "음식을 화면에 담아주세요"
    assert message_for("CLOSER", "LARGE", noun="음식") == "음식에 다가가세요"
    assert message_for("CLOSER", "SMALL") == "인물에게 조금 다가가세요"
    assert decide(p, RULE_ARTISTS["food_center"], noun="음식").direction == "CLOSER"


@pytest.mark.model
def test_food_detector_loads_and_runs():
    from PIL import Image
    from axis_ai.detect import FoodDetector
    out = FoodDetector().detect(Image.new("RGB", (320, 320), (128, 128, 128)))
    assert isinstance(out, list) and all(p.label in FOOD_CLASSES for p in out)
