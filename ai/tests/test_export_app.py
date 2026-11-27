import json

from axis_ai.bundle import ModelBundle
from axis_ai.export_app import build_local_json, make_parity
from axis_ai.features import compute_features
from axis_ai.guide import decide
from axis_ai.scorers.rule import RuleScorer
from conftest import make_head_data


def test_export_local_json_and_parity(tmp_path):
    _, _, model = make_head_data(tmp_path)
    b = ModelBundle.load(model)
    local = build_local_json(b, food_names=["person", "cup", "car"])
    json.dumps(local)  # JSON으로 쓸 수 있어야 한다
    assert set(local["stat"]) == {"s_left", "s_mid", "s_right"}
    assert {"sym_back", "food_center", "food_thirds"} <= set(local["rules"])
    assert len(local["rules"]["food_thirds"]) == 2 and len(local["rules"]["sym_back"]) == 1
    assert local["food_classes"] == [None, "cup", None]
    assert local["params"]["MOVE_STEP"] == 0.08 and local["version"] == "v1"
    ids = [a["id"] for a in local["artists"]]
    assert ids[:3] == ["s_left", "s_mid", "s_right"] and "food_thirds" in ids
    assert all(len(a["diagram_box"]) == 4 for a in local["artists"])

    cases = make_parity(b, n_person=6, n_food=2)
    assert len(cases) == 8
    c = cases[0]
    p = c["_person"]
    assert c["features"]["cx"] == compute_features(p).cx
    assert c["decide"]["s_mid"]["direction"] == decide(p, b.scorers["stat"]["s_mid"]).direction
    food = cases[-1]
    assert food["person"]["label"] != "person" and set(food["decide"]) == {"food_center", "food_thirds"}
    assert food["decide"]["food_center"]["message"] == decide(food["_person"], _food(local), "음식").message


def _food(local):
    r = local["rules"]["food_center"][0]
    return RuleScorer("food_center", r["target"], r["spread"])
