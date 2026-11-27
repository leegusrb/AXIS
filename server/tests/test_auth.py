import sqlite3

from fastapi.testclient import TestClient
from test_api import FakeCatalog, client, jpeg, raw_dir, settings  # noqa: F401 (픽스처)

from axis_server.main import create_app
from axis_server.pipeline import FakePipeline


def make_client(settings, raw_dir):
    return TestClient(create_app(settings, FakePipeline(), FakeCatalog(raw_dir)))


def _token(client, name, pw="secret1"):
    r = client.post("/auth/signup", json={"username": name, "password": pw})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def _h(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_signup_login_me(client):
    r = client.post("/auth/signup", json={"username": "hyun", "password": "secret1"})
    assert r.status_code == 200 and r.json()["username"] == "hyun"
    assert client.post("/auth/signup", json={"username": "hyun", "password": "secret1"}).status_code == 409
    assert client.post("/auth/login", json={"username": "hyun", "password": "nope00"}).status_code == 401
    assert client.post("/auth/login", json={"username": "nobody", "password": "nope00"}).status_code == 401
    tok = client.post("/auth/login", json={"username": "hyun", "password": "secret1"}).json()["token"]
    assert client.get("/me", headers=_h(tok)).json()["username"] == "hyun"
    assert client.get("/me").status_code == 401
    assert client.get("/me", headers=_h("garbage")).status_code == 401


def test_signup_validation(client):
    assert client.post("/auth/signup", json={"username": "ab", "password": "secret1"}).status_code == 422
    assert client.post("/auth/signup", json={"username": "a b c", "password": "secret1"}).status_code == 422
    assert client.post("/auth/signup", json={"username": "현규", "password": "12345"}).status_code == 422
    assert client.post("/auth/signup", json={"username": "현규_01", "password": "123456"}).status_code == 200


def test_shots_are_per_user(client):
    a, b = _token(client, "aaa"), _token(client, "bbb")
    files = {"photo": ("s.jpg", jpeg(), "image/jpeg")}
    data = {"style_id": "s1", "score": "0.8", "guide_used": "true"}
    assert client.post("/shots", files=files, data=data).status_code == 401
    assert client.post("/shots", files=files, data=data, headers=_h(a)).status_code == 200
    assert len(client.get("/shots", headers=_h(a)).json()) == 1
    assert client.get("/shots", headers=_h(b)).json() == []
    assert client.get("/shots").status_code == 401


def test_token_survives_restart(settings, raw_dir):
    tok = _token(make_client(settings, raw_dir), "ccc")
    assert make_client(settings, raw_dir).get("/me", headers=_h(tok)).status_code == 200


def test_password_is_hashed(settings, client):
    _token(client, "ddd")
    con = sqlite3.connect(settings.var_dir / "axis.db")
    row = con.execute("select password_hash from users where username='ddd'").fetchone()[0]
    con.close()
    assert row.startswith("pbkdf2_sha256$") and "secret1" not in row


def test_cors_header(client):
    r = client.get("/health", headers={"Origin": "http://x"})
    assert r.headers.get("access-control-allow-origin") == "*"


def test_shots_filter_by_style(client):
    tok = _token(client, "fff")
    for sid in ("s1", "sym_back", "s1"):
        r = client.post("/shots", files={"photo": ("s.jpg", jpeg(), "image/jpeg")},
                        data={"style_id": sid, "score": "0.5", "guide_used": "true"}, headers=_h(tok))
        assert r.status_code == 200
    assert len(client.get("/shots", headers=_h(tok)).json()) == 3
    only = client.get("/shots", params={"style_id": "sym_back"}, headers=_h(tok)).json()
    assert [s["style_id"] for s in only] == ["sym_back"]
