"""확정한 사진 목록(selected.csv)을 내려받아 data/raw/와 manifest.csv를 만든다.

사진 파일은 저장소에 커밋하지 않으므로, 목록만 있으면 누구나 같은 데이터를 다시 받을 수 있다.

    python -m axis_ai.data.fetch --selected data/selected.csv --raw data/raw --manifest data/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import time
import urllib.request
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

MANIFEST_FIELDS = ["image_id", "style_id", "file", "source_url", "author", "license"]
MAX_SIDE = 1600
USER_AGENT = "AXIS-student-project"


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def save_image(data: bytes, dest: Path) -> None:
    """EXIF 방향을 적용하고 긴 변을 MAX_SIDE 이하로 줄여 JPEG로 저장한다."""
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    im.save(tmp, "JPEG", quality=90)
    tmp.replace(dest)


def fetch(selected: Path, raw_dir: Path, manifest: Path,
          download: Callable[[str], bytes] = _download, retries: int = 3, pause: float = 0.5) -> list[str]:
    """목록의 사진을 받고 manifest를 쓴다. 받지 못한 image_id 목록을 돌려준다.

    이미 받은 파일은 다시 받지 않는다. 받지 못한 사진은 manifest에서 빠진다.
    """
    with open(selected, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    failed: list[str] = []
    out: list[dict] = []
    for r in rows:
        rel = f"{r['style_id']}/{r['image_id']}.jpg"
        dest = raw_dir / rel
        if not dest.exists():
            for attempt in range(retries):
                try:
                    save_image(download(r["download_url"]), dest)
                    break
                except Exception as e:  # 네트워크·이미지 오류는 사진 단위로 기록하고 넘어간다
                    if attempt == retries - 1:
                        print(f"실패 {r['image_id']}: {type(e).__name__}: {e}")
                    else:
                        time.sleep(pause * (attempt + 1))
            time.sleep(pause)
        if dest.exists():
            out.append({"image_id": r["image_id"], "style_id": r["style_id"], "file": rel,
                        "source_url": r["source_url"], "author": r["author"], "license": r["license"]})
        else:
            failed.append(r["image_id"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(out)
    return failed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="selected.csv의 사진을 내려받아 manifest.csv를 만든다")
    p.add_argument("--selected", type=Path, default=Path("data/selected.csv"))
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    a = p.parse_args(argv)
    failed = fetch(a.selected, a.raw, a.manifest)
    print(f"완료. 받지 못한 사진 {len(failed)}장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
