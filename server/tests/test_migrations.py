import sqlite3

from axis_server.db import make_session_factory


def _old_db(path):
    con = sqlite3.connect(path)
    con.execute("create table shots (id varchar primary key, device_id varchar not null, style_id varchar,"
                " score float, guide_used boolean, file varchar, created_at datetime)")
    con.execute("insert into shots values ('x','d','wide_full',0.5,1,'x.jpg','2026-11-25 10:00:00')")
    con.execute("create table inference_logs (id integer primary key, created_at datetime, style_id varchar,"
                " scorer varchar, direction varchar, score float, confidence float, decode_ms int,"
                " pose_ms int, guide_ms int, total_ms int)")
    con.commit()
    con.close()


def test_old_db_is_stamped_and_upgraded(tmp_path):
    old = tmp_path / "axis.db"
    _old_db(old)
    make_session_factory(old)
    con = sqlite3.connect(old)
    assert con.execute("select count(*) from shots").fetchone()[0] == 1
    cols = {r[1]: r for r in con.execute("pragma table_info(shots)")}
    assert "user_id" in cols and cols["device_id"][3] == 0  # notnull 해제
    assert con.execute("select count(*) from sqlite_master where name='users'").fetchone()[0] == 1
    head = con.execute("select version_num from alembic_version").fetchone()[0]
    con.close()
    make_session_factory(old)  # 두 번 불러도 그대로
    assert sqlite3.connect(old).execute("select version_num from alembic_version").fetchone()[0] == head


def test_fresh_db_gets_all_tables(tmp_path):
    db = tmp_path / "new.db"
    make_session_factory(db)
    names = {r[0] for r in sqlite3.connect(db).execute("select name from sqlite_master where type='table'")}
    assert {"shots", "inference_logs", "users", "alembic_version"} <= names


def test_styles_synced_with_model_version(tmp_path):
    from fastapi.testclient import TestClient
    from test_api import FakeCatalog

    from axis_server.config import Settings
    from axis_server.main import create_app
    from axis_server.pipeline import FakePipeline

    raw = tmp_path / "raw"
    raw.mkdir()
    settings = Settings(var_dir=tmp_path / "var", raw_dir=raw)
    TestClient(create_app(settings, FakePipeline(), FakeCatalog(raw)))
    con = sqlite3.connect(settings.var_dir / "axis.db")
    rows = {r[0]: r[1:] for r in con.execute("select id, category_id, kind, model_version from styles")}
    cats = dict(con.execute("select id, name from categories").fetchall())
    assert cats == {"person": "인물", "food": "음식"}
    assert rows["sym_back"] == ("person", "rule", "rule-v0")
    assert rows["food_center"][0] == "food"
    assert rows["s1"] == ("person", "learned", "fake-0")
    con.close()
    # 다시 켜도 중복 없이 갱신된다
    TestClient(create_app(settings, FakePipeline(), FakeCatalog(raw)))
    con = sqlite3.connect(settings.var_dir / "axis.db")
    assert con.execute("select count(*) from styles where id='s1'").fetchone()[0] == 1
