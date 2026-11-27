import json

import pytest

from axis_ai.bundle import ModelBundle
from axis_ai.head.train import train_heads
from conftest import FakeEmbedder, make_head_data


@pytest.fixture(scope="module")
def head_data(tmp_path_factory):
    proc, raw, model = make_head_data(tmp_path_factory.mktemp("head"))
    train_heads(proc, raw, model, FakeEmbedder(), aug=4, epochs=20)
    return proc, raw, model


def test_train_heads_writes_files_and_history(head_data):
    _, _, model = head_data
    assert sorted(p.name for p in (model / "head").glob("*.pt")) == ["s_left.pt", "s_mid.pt", "s_right.pt"]
    hist = json.loads((model / "head_history.json").read_text())
    assert set(hist) == {"s_left", "s_mid", "s_right"}
    assert hist["s_mid"][-1]["epoch"] == 20 and "val_mse" in hist["s_mid"][-1]


def test_bundle_loads_heads(head_data, tmp_path):
    _, _, model = head_data
    b = ModelBundle.load(model)
    assert set(b.heads) == {"s_left", "s_mid", "s_right"} and b.heads["s_mid"].style_id == "s_mid"


def test_bundle_without_heads(tmp_path):
    _, _, model = make_head_data(tmp_path, styles=("s_mid",))
    assert ModelBundle.load(model).heads == {}
