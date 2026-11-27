import io

import pytest
from axis_ai.features import select_subject
from axis_ai.guide import decide
from axis_ai.scorers import RULE_PRESETS
from conftest import make_person
from fastapi.testclient import TestClient
from PIL import Image

from axis_server.__main__ import build_components
from axis_server.config import Settings
from axis_server.main import BundleCatalog, create_app
from axis_server.pipeline import InvalidImageError, RealPipeline


def jpeg(size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 90, 90)).save(buf, "JPEG")
    return buf.getvalue()


class FakeDetector:
    def __init__(self, persons):
        self.persons = persons
        self.calls = 0

    def detect(self, img):
        self.calls += 1
        return self.persons


PERSON = make_person(cx=0.3, top=0.15, w=0.3, h=0.6)
RULES = {"rule": RULE_PRESETS}


def test_analyze_matches_decide_and_times_stages():
    pipe = RealPipeline({"rule": RULE_PRESETS}, "rule-v0", FakeDetector([PERSON]))
    res = pipe.analyze(jpeg(), "thirds_upper", "rule")
    assert res.decision == decide(select_subject([PERSON]), RULE_PRESETS["thirds_upper"])
    assert set(res.timing_ms) == {"decode", "pose", "guide", "depth", "embed"}
    assert all(isinstance(v, int) and v >= 0 for v in res.timing_ms.values())


def test_no_person_gives_no_subject():
    pipe = RealPipeline({"rule": RULE_PRESETS}, "rule-v0", FakeDetector([]))
    assert pipe.analyze(jpeg(), "wide_full", "rule").decision.direction == "NO_SUBJECT"


def test_bad_bytes_is_invalid_image():
    pipe = RealPipeline({"rule": RULE_PRESETS}, "rule-v0", FakeDetector([PERSON]))
    with pytest.raises(InvalidImageError):
        pipe.analyze(b"\xff\xd8\xff garbage", "wide_full", "rule")


def test_missing_style_in_scorer_is_key_error():
    pipe = RealPipeline({"rule": RULE_PRESETS}, "rule-v0", FakeDetector([PERSON]))
    with pytest.raises(KeyError):
        pipe.analyze(jpeg(), "nope", "rule")


def test_available_scorers_excludes_empty_kinds():
    pipe = RealPipeline({"stat": RULE_PRESETS, "rank": {}}, "v", FakeDetector([]))
    assert pipe.available_scorers == {"stat"}


def test_warmup_runs_detector_once_with_black_image():
    det = FakeDetector([])
    RealPipeline({"rule": RULE_PRESETS}, "v", det).warmup()
    assert det.calls == 1


def test_bundle_catalog_and_real_pipeline_end_to_end(bundle, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    cat = BundleCatalog(bundle, raw)
    styles = cat.styles()
    assert [s["id"] for s in styles] == ["s_left", "s_mid", "s_right"]
    assert all(len(s["samples"]) == 3 and set(s["samples"][0]) == {"file", "author", "license"} for s in styles)
    assert all(len(s["diagram_box"]) == 4 for s in styles)
    assert len(cat.taste_photos()) == 12
    ranked = cat.recommend(["s_left_10", "s_left_11"])
    assert ranked[0][0] == "s_left" and abs(sum(p for _, p in ranked) - 100) < 1e-6
    with pytest.raises(KeyError):
        cat.recommend(["nope"])
    pipe = RealPipeline(bundle.scorers, bundle.version, FakeDetector([PERSON]))
    assert pipe.available_scorers == {"stat"} and pipe.model_version == "v1"
    client = TestClient(create_app(Settings(var_dir=tmp_path / "var", raw_dir=raw), pipe, cat))
    r = client.post("/analyze", files={"frame": ("f.jpg", jpeg(), "image/jpeg")}, data={"style_id": "s_mid"})
    assert r.status_code == 200 and r.json()["model_version"] == "v1"
    assert client.post("/recommend", json={"liked_photo_ids": ["nope"]}).status_code == 404


def test_build_components_rule_defaults_to_rule_scorer(tmp_path):
    settings = Settings(mode="rule", var_dir=tmp_path / "var")
    pipe, cat, s = build_components(settings, detector_factory=lambda: FakeDetector([PERSON]),
                                    food_detector_factory=lambda: FakeDetector([]),
                                    depth_factory=lambda: None)
    assert s.default_scorer == "rule" and settings.default_scorer == "stat"
    client = TestClient(create_app(s, pipe, cat))
    assert client.get("/health").json() == {"status": "ok", "model_version": "rule-v0", "scorers": ["rule"]}
    r = client.post("/analyze", files={"frame": ("f.jpg", jpeg(), "image/jpeg")}, data={"style_id": "wide_full"})
    assert r.status_code == 200 and r.json()["scorer"] == "rule"


def test_build_components_model_requires_models_dir(tmp_path):
    with pytest.raises(SystemExit, match="AXIS_MODELS_DIR"):
        build_components(Settings(mode="model", models_dir=None, var_dir=tmp_path))


def test_bundle_catalog_inconsistency_is_runtime_error_not_key_error(bundle, tmp_path):
    cat = BundleCatalog(bundle, tmp_path)
    del bundle.centroids["s_mid"]  # 번들 내부 불일치
    with pytest.raises(RuntimeError):
        cat.recommend(["s_left_10"])
    with pytest.raises(KeyError):  # 모르는 사진 id는 그대로 KeyError
        cat.recommend(["nope"])
    client = TestClient(create_app(Settings(var_dir=tmp_path / "var", raw_dir=tmp_path),
                                   RealPipeline(bundle.scorers, "v1", FakeDetector([])), cat),
                        raise_server_exceptions=False)
    assert client.post("/recommend", json={"liked_photo_ids": ["s_left_10"]}).status_code == 500
    assert client.post("/recommend", json={"liked_photo_ids": ["nope"]}).status_code == 404


def test_detector_calls_are_serialized():
    import threading
    import time

    class Probe:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        def detect(self, img):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            self.active -= 1
            return []

    det = Probe()
    pipe = RealPipeline({"rule": RULE_PRESETS}, "v", det)
    ts = [threading.Thread(target=pipe.analyze, args=(jpeg(), "wide_full", "rule")) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert det.max_active == 1


def test_check_default_scorer_fails_fast(bundle, tmp_path):
    from axis_server.__main__ import check_default_scorer

    pipe = RealPipeline(bundle.scorers, "v1", FakeDetector([]))  # stat만 있음
    check_default_scorer(pipe, Settings())  # 기본 stat: 통과
    with pytest.raises(SystemExit) as e:
        check_default_scorer(pipe, Settings(default_scorer="rank"))
    assert "rank" in str(e.value) and "stat" in str(e.value)


def test_rule_artist_uses_rule_scorer_for_any_kind():
    from axis_ai.artists import RULE_ARTISTS
    pipe = RealPipeline({"stat": {}}, "v", FakeDetector([PERSON]))
    res = pipe.analyze(jpeg(), "sym_back", "stat")
    assert res.scorer_used == "rule"
    assert res.decision == decide(select_subject([PERSON]), RULE_ARTISTS["sym_back"])


def test_food_style_ignores_person():
    pose, food = FakeDetector([PERSON]), FakeDetector([])
    pipe = RealPipeline({"stat": {}}, "v", pose, food_detector=food)
    res = pipe.analyze(jpeg(), "food_center", "stat", category="food")
    assert res.decision.direction == "NO_SUBJECT"
    assert res.decision.message == "음식을 화면에 담아주세요"
    assert pose.calls == 0 and food.calls == 1


def test_food_style_guides_food_and_without_detector_no_subject():
    from axis_ai.detect import boxes_to_food
    import numpy as np
    cake = boxes_to_food(np.array([[0.3, 0.3, 0.7, 0.7]]), np.array([0.9]), ["cake"])
    pipe = RealPipeline({"stat": {}}, "v", FakeDetector([PERSON]), food_detector=FakeDetector(cake))
    assert pipe.analyze(jpeg(), "food_center", "stat", category="food").decision.message.startswith("음식에")
    no_food = RealPipeline({"stat": {}}, "v", FakeDetector([PERSON]))
    assert no_food.analyze(jpeg(), "food_center", "stat", category="food").decision.direction == "NO_SUBJECT"


def test_api_passes_food_category(bundle, tmp_path):
    food = FakeDetector([])
    pipe = RealPipeline(bundle.scorers, "v", FakeDetector([PERSON]), food_detector=food)
    client = TestClient(create_app(Settings(var_dir=tmp_path / "var", raw_dir=tmp_path), pipe,
                                   BundleCatalog(bundle, tmp_path)))
    r = client.post("/analyze", files={"frame": ("f.jpg", jpeg(), "image/jpeg")}, data={"style_id": "food_thirds"})
    assert r.status_code == 200 and r.json()["direction"] == "NO_SUBJECT" and food.calls == 1


def _tilted(p, dy):
    from dataclasses import replace
    kp = p.keypoints.copy()
    kp[6, 1] += dy
    return replace(p, keypoints=kp)


def test_ok_with_tilt_keeps_ok_and_changes_message(bundle, tmp_path):
    from axis_ai.guide import decide as _decide
    # s_mid 학습 분포 한가운데 인물 -> OK가 나와야 한다
    p = make_person(cx=0.5, top=0.2, w=0.3, h=0.6)
    assert _decide(p, bundle.scorers["stat"]["s_mid"]).direction == "OK"
    pipe = RealPipeline(bundle.scorers, "v", FakeDetector([_tilted(p, 0.06)]))
    res = pipe.analyze(jpeg(), "s_mid", "stat")
    assert res.decision.direction == "OK" and res.rotate == "CW"
    assert res.decision.message == "좋아요! 수평만 맞추면 완벽해요"
    assert res.rotation_deg > 8 and res.subject.label == "person"
    level = RealPipeline(bundle.scorers, "v", FakeDetector([p])).analyze(jpeg(), "s_mid", "stat")
    assert level.rotate is None and level.decision.message == "좋아요! 지금 찍으세요"


def test_guide_result_has_subject_rotation_category(bundle, tmp_path):
    pipe = RealPipeline(bundle.scorers, "v", FakeDetector([_tilted(PERSON, -0.06)]))
    client = TestClient(create_app(Settings(var_dir=tmp_path / "var", raw_dir=tmp_path), pipe,
                                   BundleCatalog(bundle, tmp_path)))
    j = client.post("/analyze", files={"frame": ("f.jpg", jpeg(), "image/jpeg")},
                    data={"style_id": "s_mid"}).json()
    assert j["rotate"] == "CCW" and j["rotation_deg"] < -8 and j["category"] == "person"
    sub = j["subject"]
    assert sub["class_name"] == "person" and set(sub) == {"class_name", "cx", "cy", "w", "h", "area", "conf"}
    assert sub["cx"] == pytest.approx(PERSON.box[0] + PERSON.box[2] / 2)
    none = RealPipeline(bundle.scorers, "v", FakeDetector([]))
    client = TestClient(create_app(Settings(var_dir=tmp_path / "var2", raw_dir=tmp_path), none,
                                   BundleCatalog(bundle, tmp_path)))
    j = client.post("/analyze", files={"frame": ("f.jpg", jpeg(), "image/jpeg")},
                    data={"style_id": "s_mid"}).json()
    assert j["subject"] is None and j["rotate"] is None and j["rotation_deg"] is None


class FakeDepth:
    """박스 영역만 1, 나머지 0인 깊이 지도."""
    DIFF = 1.0

    def __init__(self):
        self.calls = 0

    def estimate(self, img):
        import numpy as np
        self.calls += 1
        d = np.zeros((48, 64), np.float32)
        x, y, w, h = PERSON.box
        d[round(y * 48):round((y + h) * 48), round(x * 64):round((x + w) * 64)] = 1.0
        return d


def test_depth_preview_only_when_asked():
    depth = FakeDepth()
    pipe = RealPipeline(RULES, "v", FakeDetector([PERSON]), depth=depth)
    res = pipe.analyze(jpeg(), "thirds_upper", "rule")
    assert res.depth_preview is None and res.depth is None and depth.calls == 0
    assert res.timing_ms["depth"] == 0
    res = pipe.analyze(jpeg(), "thirds_upper", "rule", depth_preview=True)
    assert res.depth_preview and res.depth.diff == pytest.approx(FakeDepth.DIFF, abs=1e-3)
    assert depth.calls == 1


def test_depth_preview_null_without_estimator():
    res = RealPipeline(RULES, "v", FakeDetector([PERSON])).analyze(
        jpeg(), "thirds_upper", "rule", depth_preview=True)
    assert res.depth_preview is None and res.decision.direction


def test_api_depth_fields(bundle, tmp_path):
    pipe = RealPipeline(bundle.scorers, "v", FakeDetector([PERSON]), depth=FakeDepth())
    client = TestClient(create_app(Settings(var_dir=tmp_path / "var", raw_dir=tmp_path), pipe,
                                   BundleCatalog(bundle, tmp_path)))
    send = lambda **d: client.post("/analyze", files={"frame": ("f.jpg", jpeg(), "image/jpeg")},  # noqa: E731
                                   data={"style_id": "s_mid", **d}).json()
    j = send()
    assert j["depth"] is None and j["depth_preview"] is None and j["timing_ms"]["depth"] == 0
    j = send(depth_preview="true")
    assert j["depth"]["diff"] == pytest.approx(1.0, abs=1e-3) and len(j["depth_preview"]) > 20
    assert set(j["timing_ms"]) == {"decode", "pose", "guide", "depth", "embed", "total"}


class FakeHead:
    def __init__(self, c):
        import numpy as np
        self.c = np.array(c, dtype=np.float32)
        self.seen = None

    def predict(self, x):
        self.seen = x
        return self.c


class FakeEmbed:
    def __init__(self):
        self.calls = 0

    def embed(self, img):
        import numpy as np
        self.calls += 1
        return np.zeros(576, np.float32)


def test_head_falls_back_for_rule_artist(bundle):
    head, emb = FakeHead([-0.2, 0, 0, -10 / 30]), FakeEmbed()
    pipe = RealPipeline(bundle.scorers, "v", FakeDetector([PERSON]), heads={"s_mid": head}, embedder=emb,
                        depth=FakeDepth())
    assert "head" in pipe.available_scorers
    assert pipe.analyze(jpeg(), "sym_back", "head").scorer_used == "rule"
    assert pipe.analyze(jpeg(), "s_left", "head").scorer_used == "stat"  # head 없음
    assert emb.calls == 0
    res = pipe.analyze(jpeg(), "s_mid", "head")
    assert res.scorer_used == "head" and emb.calls == 1 and head.seen.shape == (588,)
    assert res.decision.direction == "RIGHT" and res.rotate == "CW" and res.rotation_deg == pytest.approx(10)
    assert res.depth is not None and res.timing_ms["embed"] >= 0
    stat = bundle.scorers["stat"]["s_mid"]
    from axis_ai.features import compute_features
    assert res.decision.score == pytest.approx(stat.score(compute_features(PERSON)))


def test_no_head_without_embedder(bundle):
    pipe = RealPipeline(bundle.scorers, "v", FakeDetector([PERSON]), heads={"s_mid": FakeHead([0, 0, 0, 0])})
    assert "head" not in pipe.available_scorers


def test_api_head_request_without_heads_uses_stat(bundle, tmp_path):
    pipe = RealPipeline(bundle.scorers, "v", FakeDetector([PERSON]))
    client = TestClient(create_app(Settings(var_dir=tmp_path / "var", raw_dir=tmp_path), pipe,
                                   BundleCatalog(bundle, tmp_path)))
    r = client.post("/analyze", files={"frame": ("f.jpg", jpeg(), "image/jpeg")},
                    data={"style_id": "s_mid", "scorer": "head"})
    assert r.status_code == 200 and r.json()["scorer"] == "stat"
    r = client.post("/analyze", files={"frame": ("f.jpg", jpeg(), "image/jpeg")},
                    data={"style_id": "s_mid", "scorer": "nope"})
    assert r.status_code == 400


def test_head_mode_without_person_is_no_subject(bundle):
    pipe = RealPipeline(bundle.scorers, "v", FakeDetector([]), heads={"s_mid": FakeHead([0, 0, 0, 0])},
                        embedder=FakeEmbed())
    res = pipe.analyze(jpeg(), "s_mid", "head")
    assert res.decision.direction == "NO_SUBJECT" and res.scorer_used == "head"


def test_unavailable_scorer_falls_back_to_default(settings_rule_client):
    client = settings_rule_client
    for sc in ("stat", "rank", "head"):
        r = client.post("/analyze", files={"frame": ("f.jpg", jpeg(), "image/jpeg")},
                        data={"style_id": "wide_full", "scorer": sc})
        assert r.status_code == 200 and r.json()["scorer"] == "rule", (sc, r.text)


@pytest.fixture
def settings_rule_client(tmp_path):
    settings = Settings(mode="rule", var_dir=tmp_path / "var")
    pipe, cat, s = build_components(settings, detector_factory=lambda: FakeDetector([PERSON]),
                                    food_detector_factory=lambda: FakeDetector([]), depth_factory=lambda: None)
    return TestClient(create_app(s, pipe, cat))


def test_depth_and_embed_times_are_logged(bundle, tmp_path):
    import sqlite3
    pipe = RealPipeline(bundle.scorers, "v", FakeDetector([PERSON]), depth=FakeDepth())
    var = tmp_path / "var"
    client = TestClient(create_app(Settings(var_dir=var, raw_dir=tmp_path), pipe, BundleCatalog(bundle, tmp_path)))
    import time as _t
    orig = FakeDepth.estimate
    FakeDepth.estimate = lambda self, img: (_t.sleep(0.02), orig(self, img))[1]
    try:
        client.post("/analyze", files={"frame": ("f.jpg", jpeg(), "image/jpeg")},
                    data={"style_id": "s_mid", "depth_preview": "true"})
    finally:
        FakeDepth.estimate = orig
    row = sqlite3.connect(var / "axis.db").execute("select depth_ms, embed_ms from inference_logs").fetchone()
    assert row[0] >= 15 and row[1] == 0
