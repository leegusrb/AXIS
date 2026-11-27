import csv
import io

from PIL import Image

from axis_ai.data.fetch import MANIFEST_FIELDS, fetch


def _jpeg(w=2400, h=1200):
    b = io.BytesIO()
    Image.new("RGB", (w, h), (200, 50, 50)).save(b, "JPEG")
    return b.getvalue()


def _selected(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "style_id", "download_url", "source_url", "author", "license"])
        w.writerows(rows)


def test_fetch_writes_resized_images_and_manifest(tmp_path):
    sel = tmp_path / "selected.csv"
    _selected(sel, [["a1", "wide_full", "http://x/a1", "http://src/a1", "Kim, Lee", "CC BY 2.0"],
                    ["b2", "tight_close", "http://x/b2", "http://src/b2", "", "CC0 1.0"]])
    calls = []
    failed = fetch(sel, tmp_path / "raw", tmp_path / "manifest.csv",
                   download=lambda u: calls.append(u) or _jpeg(), pause=0)
    assert failed == []
    im = Image.open(tmp_path / "raw" / "wide_full" / "a1.jpg")
    assert max(im.size) == 1600
    rows = list(csv.DictReader(open(tmp_path / "manifest.csv", encoding="utf-8")))
    assert list(rows[0].keys()) == MANIFEST_FIELDS
    assert rows[0] == {"image_id": "a1", "style_id": "wide_full", "file": "wide_full/a1.jpg",
                       "source_url": "http://src/a1", "author": "Kim, Lee", "license": "CC BY 2.0"}
    # 이미 받은 파일은 다시 받지 않는다
    fetch(sel, tmp_path / "raw", tmp_path / "manifest.csv", download=lambda u: calls.append(u) or _jpeg(), pause=0)
    assert len(calls) == 2


def test_failed_download_is_reported_and_left_out(tmp_path):
    sel = tmp_path / "selected.csv"
    _selected(sel, [["ok", "wide_full", "http://x/ok", "s", "a", "CC0 1.0"],
                    ["bad", "wide_full", "http://x/bad", "s", "a", "CC0 1.0"],
                    ["junk", "wide_full", "http://x/junk", "s", "a", "CC0 1.0"]])

    def dl(u):
        if u.endswith("bad"):
            raise OSError("boom")
        if u.endswith("junk"):
            return b"not an image"
        return _jpeg(300, 500)

    failed = fetch(sel, tmp_path / "raw", tmp_path / "manifest.csv", download=dl, pause=0)
    assert failed == ["bad", "junk"]
    rows = list(csv.DictReader(open(tmp_path / "manifest.csv", encoding="utf-8")))
    assert [r["image_id"] for r in rows] == ["ok"]
    assert not list((tmp_path / "raw" / "wide_full").glob("*.part"))
