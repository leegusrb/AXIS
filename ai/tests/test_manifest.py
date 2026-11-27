import random
from collections import Counter

import pytest
from PIL import Image

from axis_ai.data.manifest import (
    PhotoRecord, assign_splits, license_allowed, load_manifest, split_warnings, validate,
)


def rec(image_id, style="s1", file=None, author="a", lic="CC0"):
    return PhotoRecord(image_id, style, file or f"{style}/{image_id}.jpg", "http://x", author, lic)


def save(raw, r, size=(500, 600), color=(10, 20, 30)):
    p = raw / r.file
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(p)


@pytest.mark.parametrize("lic", [
    "CC0", "CC0 1.0", "cc0", "PDM", "Public Domain Mark 1.0", "public  domain mark",
    "CC BY 4.0", "CC BY", "cc by 2.0", "CC BY-SA 2.0", "CC BY-SA", " CC  BY-SA   4.0 ",
    "Unsplash License", "Pexels License", "consent", "Consent",
    "CC BY-SA 4.0 International", "CC BY 2.0 Generic", "CC BY-SA 3.0 Unported", "CC BY 3.0 US",
    "CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0", "CC0 1.0 Universal", "cc-by-sa-2.0",
])
def test_accepts_allowed(lic):
    assert license_allowed(lic)


@pytest.mark.parametrize("lic", [
    "", "   ", None, "All rights reserved", "CC BY-NC 4.0", "CC BY-ND", "CC BY-NC-SA 4.0",
    "CC BY-NC-ND", "CC BY-SA-NC", "CC BYX", "CC BY 4.0 NC", "CC BY-ND 2.0", "CC0X", "PDMX", "consents",
    "CC BY-NC 4.0 International", "CC-BY-NC-SA-4.0", "CC-BY-NC-4.0", "CC-BY-ND-4.0", "CC BY-ND 4.0 International",
    "CC BY-NC-SA 3.0 Unported", "CC BY 4.0 ND", "CC-BY-SA-NC-4.0", "CC BY 4.0 International NC",
])
def test_rejects_disallowed(lic):
    assert not license_allowed(lic)


def test_load_manifest(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("image_id,style_id,file,source_url,author,license\n"
                 "a,s1,s1/a.jpg,http://u,Jane,CC BY 4.0\n", encoding="utf-8")
    assert load_manifest(f) == [PhotoRecord("a", "s1", "s1/a.jpg", "http://u", "Jane", "CC BY 4.0")]


def test_rejects_disallowed_license(tmp_path):
    r = rec("bad", lic="All rights reserved")
    save(tmp_path, r)
    ok, problems = validate([r], tmp_path)
    assert ok == []
    assert any("bad" in p for p in problems)


def test_rejects_missing_file_small_and_duplicate(tmp_path):
    good, missing, small, dup1, dup2 = rec("good"), rec("missing"), rec("small"), rec("d1"), rec("d2")
    save(tmp_path, good, color=(1, 2, 3))
    save(tmp_path, small, size=(399, 900))
    save(tmp_path, dup1, color=(9, 9, 9))
    (tmp_path / dup2.file).write_bytes((tmp_path / dup1.file).read_bytes())
    ok, problems = validate([good, missing, small, dup1, dup2], tmp_path)
    assert [r.image_id for r in ok] == ["good", "d1"]
    rejected = [p for p in problems if not p.startswith("경고")]
    assert len(rejected) == 3
    for name in ("missing", "small", "d2"):
        assert any(name in p for p in rejected)


def test_rejects_unopenable_image(tmp_path):
    r = rec("junk")
    (tmp_path / "s1").mkdir()
    (tmp_path / r.file).write_bytes(b"not an image")
    ok, problems = validate([r], tmp_path)
    assert ok == [] and any("junk" in p for p in problems)


def test_exif_rotation_applied_before_size_check(tmp_path):
    r = rec("rot")
    p = tmp_path / r.file
    p.parent.mkdir()
    img = Image.new("RGB", (800, 300), (5, 5, 5))  # 회전 후 300x800 → 짧은 변 300 < 400
    exif = Image.Exif()
    exif[0x0112] = 6
    img.save(p, exif=exif)
    ok, problems = validate([r], tmp_path)
    assert ok == [] and any("rot" in x for x in problems)


def test_warns_author_over_quarter(tmp_path):
    recs = [rec(f"a{i}", author="Same") for i in range(3)] + [rec(f"b{i}", author=f"o{i}") for i in range(5)]
    for i, r in enumerate(recs):
        save(tmp_path, r, color=(i, i, i))
    ok, problems = validate(recs, tmp_path)
    assert len(ok) == 8
    assert any("Same" in p for p in problems)


def test_warns_style_under_50(tmp_path):
    r = rec("one")
    save(tmp_path, r)
    ok, problems = validate([r], tmp_path)
    assert len(ok) == 1 and any("s1" in p and "50" in p for p in problems)


def _records(n=60):
    return [rec(f"{s}_{i:03d}", style=s) for s in ("s1", "s2", "s3") for i in range(n)]


def test_splits_deterministic_and_sized():
    recs = _records()
    s = assign_splits(recs)
    for style in ("s1", "s2", "s3"):
        c = Counter(r.split for r in s if r.style_id == style)
        assert c == {"taste": 4, "test": 10, "train": 46}
    shuffled = recs[:]
    random.Random(1).shuffle(shuffled)
    assert assign_splits(shuffled) == s


def test_splits_small_style_all_taste_then_test():
    s = assign_splits([rec(f"x{i}") for i in range(12)])
    assert Counter(r.split for r in s) == {"taste": 4, "test": 8}


def test_rejects_duplicate_image_id(tmp_path):
    a, b = rec("same", file="s1/1.jpg"), rec("same", file="s1/2.jpg")
    save(tmp_path, a, color=(1, 1, 1))
    save(tmp_path, b, color=(2, 2, 2))
    ok, problems = validate([a, b], tmp_path)
    assert ok == [a]
    assert any("same" in p and "s1/2.jpg" in p for p in problems)


def test_warns_when_train_under_ten():
    s = assign_splits([rec(f"x{i}") for i in range(20)] + [rec(f"y{i}", style="s2") for i in range(30)])
    w = split_warnings(s)
    assert len(w) == 1 and "s1" in w[0] and "6" in w[0]
