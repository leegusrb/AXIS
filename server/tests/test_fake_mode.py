import pytest
from fastapi.testclient import TestClient

from axis_server.__main__ import build_components
from axis_server.config import Settings
from axis_server.main import create_app


@pytest.fixture
def client(tmp_path):
    pipe, cat, s = build_components(Settings(mode="fake", var_dir=tmp_path / "var"))
    return TestClient(create_app(s, pipe, cat))


def test_health_and_styles(client):
    assert client.get("/health").json()["model_version"] == "fake-0"
    styles = client.get("/styles", params={"category": "person"}).json()
    styles = [s for s in styles if s["kind"] == "learned"]
    assert [(s["id"], s["name"]) for s in styles] == [
        ("wide_full", "작가 A — 여백과 인물 중심"), ("tight_close", "작가 C — 꽉 찬 클로즈업"),
        ("thirds_upper", "작가 D — 삼분할 상반신")]
    assert all(s["diagram_box"] for s in styles)
    assert all(len(s["samples"]) == 3 and s["samples"][0]["license"] == "CC0" for s in styles)
    assert client.get(styles[0]["samples"][0]["url"]).status_code == 200


def test_taste_photos_serve_images(client):
    photos = client.get("/taste/photos").json()
    assert len(photos) == 12 and photos[0]["id"] == "taste_01"
    r = client.get(photos[0]["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_recommend_is_deterministic_and_follows_likes(client):
    ids = [p["id"] for p in client.get("/taste/photos").json()]
    # taste_01, 04, 07, 10 은 첫 스타일(wide_full)
    body = {"liked_photo_ids": [ids[0], ids[3], ids[6]]}
    a = client.post("/recommend", json=body).json()
    assert a == client.post("/recommend", json=body).json()
    assert a[0]["style_id"] == "wide_full"
    assert abs(sum(x["score"] for x in a) - 100) < 1e-6
    assert client.post("/recommend", json={"liked_photo_ids": ["nope"]}).status_code == 404
