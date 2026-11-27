import io
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from axis_server.config import Settings
from axis_server.main import create_app
from axis_server.pipeline import FakePipeline


class FakeCatalog:
    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir

    def styles(self):
        return [{"id": "s1", "name": "스타일 1", "description": "설명",
                 "samples": [{"file": "a.jpg", "author": "작가", "license": "CC0"}]}]

    def taste_photos(self):
        return [{"image_id": "t1", "file": "t1.jpg"}, {"image_id": "t2", "file": "t2.jpg"}]

    def recommend(self, image_ids):
        for i in image_ids:
            if i not in {"t1", "t2"}:
                raise KeyError(i)
        return [("s1", 0.9), ("s2", 0.1)]


def jpeg(gray=100, size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("L", size, gray).convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def raw_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    (d / "a.jpg").write_bytes(jpeg())
    return d


@pytest.fixture
def settings(tmp_path, raw_dir):
    return Settings(var_dir=tmp_path / "var", raw_dir=raw_dir, max_upload_bytes=10_000,
                    max_shot_bytes=20_000)


@pytest.fixture
def client(settings, raw_dir):
    return TestClient(create_app(settings, FakePipeline(), FakeCatalog(raw_dir)))


def log_count(settings) -> int:
    con = sqlite3.connect(settings.var_dir / "axis.db")
    try:
        return con.execute("select count(*) from inference_logs").fetchone()[0]
    finally:
        con.close()


def analyze(client, data=None, style_id="s1", **extra):
    return client.post("/analyze", files={"frame": ("f.jpg", data if data is not None else jpeg(), "image/jpeg")},
                       data={"style_id": style_id, **extra})


def test_health_reports_model_and_scorers(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "fake-0", "scorers": ["stat"]}


def test_analyze_returns_guide_result_and_logs(client, settings):
    assert log_count(settings) == 0
    r = analyze(client)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"direction", "magnitude", "message", "score", "score_delta", "confidence",
                         "subject_box", "target_box", "scorer", "model_version", "timing_ms",
                         "category", "subject", "rotation_deg", "rotate", "depth", "depth_preview"}
    assert body["direction"] == "DOWN"  # 밝기 100 -> 4번째 방향
    assert body["message"] == "카메라를 아래로 조금 옮기세요" or body["message"] == "카메라를 아래로 옮기세요"
    assert body["magnitude"] in ("SMALL", "LARGE")
    assert len(body["subject_box"]) == 4 and len(body["target_box"]) == 4
    assert body["scorer"] == "stat" and body["model_version"] == "fake-0"
    assert set(body["timing_ms"]) == {"decode", "pose", "guide", "depth", "embed", "total"}
    assert all(isinstance(v, int) and v >= 0 for v in body["timing_ms"].values())
    assert log_count(settings) == 1
    assert analyze(client).status_code == 200
    assert log_count(settings) == 2


def test_analyze_deterministic_and_no_subject_has_null_boxes(client):
    a, b = analyze(client).json(), analyze(client).json()
    a.pop("timing_ms"), b.pop("timing_ms")
    assert a == b
    r =analyze(client, jpeg(250)).json()
    assert r["direction"] == "NO_SUBJECT"
    assert r["subject_box"] is None and r["target_box"] is None


def test_analyze_rejects_non_jpeg(client, settings):
    assert analyze(client, b"\x89PNG\r\n not a jpeg").status_code == 400
    assert log_count(settings) == 0


def test_analyze_rejects_corrupt_jpeg(client, settings):
    assert analyze(client, b"\xff\xd8\xff garbage").status_code == 400
    assert log_count(settings) == 0


def test_analyze_rejects_oversize(client):
    assert analyze(client, b"\xff\xd8\xff" + b"0" * 10_001).status_code == 413


def test_analyze_unknown_style(client):
    assert analyze(client, style_id="nope").status_code == 404


def test_analyze_unavailable_scorer(client):
    # 알려진 방식이지만 이 서버에 없으면 기본 방식으로 대신하고, 모르는 이름은 400
    r = analyze(client, scorer="rank")
    assert r.status_code == 200 and r.json()["scorer"] == "stat"
    assert analyze(client, scorer="nope").status_code == 400


def test_analyze_explicit_available_scorer(client):
    r = analyze(client, scorer="stat")
    assert r.status_code == 200 and r.json()["scorer"] == "stat"


def test_styles_urls_and_taste_photos_hide_style(client):
    s = client.get("/styles").json()
    assert s[0]["samples"] == [{"url": "/media/raw/a.jpg", "author": "작가", "license": "CC0"}]
    assert client.get(s[0]["samples"][0]["url"]).status_code == 200
    photos = client.get("/taste/photos").json()
    assert photos == [{"id": "t1", "url": "/media/raw/t1.jpg"}, {"id": "t2", "url": "/media/raw/t2.jpg"}]
    assert all("style_id" not in p for p in photos)


def test_recommend_ok(client):
    r = client.post("/recommend", json={"liked_photo_ids": ["t1"]})
    assert r.status_code == 200
    assert r.json() == [{"style_id": "s1", "score": 0.9}, {"style_id": "s2", "score": 0.1}]


def test_recommend_empty_list(client):
    assert client.post("/recommend", json={"liked_photo_ids": []}).status_code == 422


def test_recommend_unknown_photo_id(client):
    assert client.post("/recommend", json={"liked_photo_ids": ["t1", "zzz"]}).status_code == 404


def token(client, name="user1"):
    r = client.post("/auth/signup", json={"username": name, "password": "secret1"})
    if r.status_code == 409:
        r = client.post("/auth/login", json={"username": name, "password": "secret1"})
    return r.json()["token"]


def shot(client, user="user1", data=None, **extra):
    return client.post("/shots", files={"photo": ("p.jpg", data if data is not None else jpeg(), "image/jpeg")},
                       data={"style_id": "s1", "score": "0.7", "guide_used": "true", **extra},
                       headers={"Authorization": f"Bearer {token(client, user)}"})


def test_shots_roundtrip_by_user(client, settings):
    a1 = shot(client, "user1")
    assert a1.status_code == 200
    a2 = shot(client, "user1")
    shot(client, "user2")
    rows = client.get("/shots", headers={"Authorization": f"Bearer {token(client, 'user1')}"}).json()
    assert [r["id"] for r in rows] == [a2.json()["id"], a1.json()["id"]]  # 최신순
    r = rows[0]
    assert set(r) == {"id", "style_id", "score", "guide_used", "url", "created_at"}
    assert r["style_id"] == "s1" and r["score"] == 0.7 and r["guide_used"] is True
    assert r["created_at"].endswith("+00:00")
    assert r["url"] == f"/media/shots/{r['id']}.jpg"
    assert (settings.var_dir / "shots" / f"{r['id']}.jpg").is_file()
    assert client.get(r["url"]).content == jpeg()
    assert client.get("/shots", headers={"Authorization": f"Bearer {token(client, 'other')}"}).json() == []


def test_shots_validation(client):
    assert shot(client, data=b"not jpeg").status_code == 400
    assert shot(client, data=b"\xff\xd8\xff" + b"0" * 20_001).status_code == 413
    # 프레임 한도(10KB)보다 큰 촬영 사진은 허용
    assert shot(client, data=b"\xff\xd8\xff" + b"0" * 15_000).status_code == 200


@pytest.mark.parametrize("path", ["../axis.db", "..%2Faxis.db", "%2e%2e/axis.db", "../../etc/passwd"])
def test_media_path_traversal_blocked(client, settings, path):
    shot(client)  # var_dir/axis.db는 존재
    assert client.get(f"/media/raw/{path}").status_code == 404
    assert client.get(f"/media/shots/{path}").status_code == 404
    assert client.get(f"/media/{path}").status_code == 404
    assert (settings.var_dir / "axis.db").exists()


def test_media_missing_file_404(client):
    assert client.get("/media/raw/missing.jpg").status_code == 404
    assert client.get("/media/shots/missing.jpg").status_code == 404


@pytest.mark.parametrize("exc", [KeyError("x"), RuntimeError("boom")])
def test_analyze_pipeline_bug_is_500_and_not_logged(settings, raw_dir, exc):
    class Broken(FakePipeline):
        def analyze(self, jpeg, style_id, scorer, **kw):
            raise exc

    c = TestClient(create_app(settings, Broken(), FakeCatalog(raw_dir)), raise_server_exceptions=False)
    assert analyze(c).status_code == 500
    assert log_count(settings) == 0


def test_shots_unknown_style(client):
    assert shot(client, style_id="nope").status_code == 404
    assert client.get("/shots", headers={"Authorization": f"Bearer {token(client)}"}).json() == []


def _big_jpeg(size=(2000, 1000)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, "JPEG")
    return buf.getvalue()


def test_media_raw_width_resizes_and_caches(client, settings, raw_dir):
    (raw_dir / "big.jpg").write_bytes(_big_jpeg())
    r = client.get("/media/raw/big.jpg?w=600")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(r.content)).size == (600, 300)
    assert (settings.var_dir / "thumbs" / "600" / "big.jpg").is_file()
    assert Image.open(io.BytesIO(client.get("/media/raw/big.jpg?w=600").content)).size == (600, 300)
    assert len(client.get("/media/raw/big.jpg").content) == len(_big_jpeg())  # w 없으면 원본


def test_media_raw_width_clamped_and_never_upscales(client, raw_dir):
    (raw_dir / "big.jpg").write_bytes(_big_jpeg())
    assert Image.open(io.BytesIO(client.get("/media/raw/big.jpg?w=10").content)).size == (64, 32)
    assert Image.open(io.BytesIO(client.get("/media/raw/big.jpg?w=99999").content)).size == (1200, 600)
    assert Image.open(io.BytesIO(client.get("/media/raw/a.jpg?w=600").content)).size == (64, 48)
    assert client.get("/media/raw/big.jpg?w=abc").status_code == 422


def test_media_raw_width_applies_exif_orientation(client, raw_dir):
    im = Image.new("RGB", (400, 200), (10, 10, 10))
    exif = im.getexif()
    exif[0x0112] = 6  # 시계 방향 90도 회전해서 보여야 함
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    (raw_dir / "rot.jpg").write_bytes(buf.getvalue())
    assert Image.open(io.BytesIO(client.get("/media/raw/rot.jpg?w=100").content)).size == (100, 200)


@pytest.mark.parametrize("path", ["../axis.db", "..%2Faxis.db", "%2e%2e/axis.db", "../../etc/passwd"])
def test_media_raw_width_path_traversal_blocked(client, settings, path):
    shot(client)
    assert client.get(f"/media/raw/{path}?w=600").status_code == 404
    assert not (settings.var_dir / "thumbs").exists()
    assert client.get("/media/raw/missing.jpg?w=600").status_code == 404


def test_categories_and_styles_filter(client):
    assert [c["id"] for c in client.get("/categories").json()] == ["person", "food"]
    food = client.get("/styles", params={"category": "food"}).json()
    assert {s["id"] for s in food} == {"food_center", "food_thirds"}
    assert all(s["kind"] == "rule" and s["samples"] == [] and s["diagram_box"] for s in food)
    person = {s["id"]: s for s in client.get("/styles", params={"category": "person"}).json()}
    assert "sym_back" in person and person["sym_back"]["name"] == "작가 B — 대칭과 배경 강조"
    assert person["s1"]["kind"] == "learned" and person["s1"]["model_version"] == "fake-0"
    assert len(client.get("/styles").json()) == len(food) + len(person)


def test_analyze_rule_artist_any_scorer(client):
    r = analyze(client, style_id="sym_back", scorer="rank")
    assert r.status_code == 200 and r.json()["scorer"] == "rule"
