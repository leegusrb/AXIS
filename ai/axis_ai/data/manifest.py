"""manifest.csv 읽기, 검증, train/test/taste 분할."""
import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps

MIN_SHORT_SIDE = 400
MIN_PER_STYLE = 50
MAX_AUTHOR_SHARE = 0.25
N_TASTE, N_TEST = 4, 10

# 정규화(소문자, 공백 하나로)한 라이선스 문자열이 이 중 하나와 통째로 맞아야 한다.
# NC/ND 변형은 어느 패턴과도 맞지 않는다.
_ALLOWED = [
    re.compile(p)
    for p in (
        r"cc0([ -].*)?",
        r"(pdm|public domain mark)( .*)?",
        r"cc[ -]by(-sa)?([ -]\d+(\.\d+)?)?( (?!nc$|nd$)(international|generic|unported|[a-z]{2}))?",
        r"unsplash license",
        r"pexels license",
        r"consent",
    )
]


@dataclass
class PhotoRecord:
    image_id: str
    style_id: str
    file: str
    source_url: str
    author: str
    license: str
    split: Literal["train", "test", "taste"] = "train"


def license_allowed(lic: str | None) -> bool:
    norm = " ".join((lic or "").lower().split())
    return any(p.fullmatch(norm) for p in _ALLOWED)


def load_manifest(csv_path: Path) -> list[PhotoRecord]:
    keys = ("image_id", "style_id", "file", "source_url", "author", "license")
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [PhotoRecord(*((row[k] or "").strip() for k in keys)) for row in csv.DictReader(f)]


def open_image(path: Path) -> Image.Image:
    """EXIF 회전을 적용한 RGB 이미지."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def validate(records: list[PhotoRecord], raw_dir: Path) -> tuple[list[PhotoRecord], list[str]]:
    """(통과 목록, 사람이 읽을 문제 목록). 거부 사유는 그대로, 경고는 '경고:'로 시작한다."""
    ok: list[PhotoRecord] = []
    problems: list[str] = []
    seen: dict[str, str] = {}
    seen_ids: set[str] = set()
    for r in records:
        if r.image_id in seen_ids:
            problems.append(f"{r.image_id}: manifest에 image_id가 중복돼요 ({r.file})")
            continue
        seen_ids.add(r.image_id)
        if not license_allowed(r.license):
            problems.append(f"{r.image_id}: 허용되지 않는 라이선스 {r.license!r}")
            continue
        path = Path(raw_dir) / r.file
        if not path.is_file():
            problems.append(f"{r.image_id}: 파일이 없어요 ({r.file})")
            continue
        try:
            img = open_image(path)
        except Exception as e:  # PIL이 다양한 예외를 던진다
            problems.append(f"{r.image_id}: 이미지를 열 수 없어요 ({type(e).__name__})")
            continue
        if min(img.size) < MIN_SHORT_SIDE:
            problems.append(f"{r.image_id}: 짧은 변이 {min(img.size)}px로 {MIN_SHORT_SIDE}px보다 작아요")
            continue
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        if digest in seen:
            problems.append(f"{r.image_id}: {seen[digest]}와 같은 사진이에요")
            continue
        seen[digest] = r.image_id
        ok.append(r)

    by_style: dict[str, list[PhotoRecord]] = defaultdict(list)
    for r in ok:
        by_style[r.style_id].append(r)
    for style, rs in sorted(by_style.items()):
        if len(rs) < MIN_PER_STYLE:
            problems.append(f"경고: 스타일 {style}의 사진이 {len(rs)}장으로 {MIN_PER_STYLE}장보다 적어요")
        for author, n in sorted(Counter(r.author for r in rs).items()):
            if n > MAX_AUTHOR_SHARE * len(rs):
                problems.append(f"경고: 스타일 {style}에서 사진가 {author}의 사진이 {n}/{len(rs)}장으로 25%를 넘어요")
    return ok, problems


def _order_key(r: PhotoRecord) -> tuple[str, str]:
    return hashlib.sha1(r.image_id.encode("utf-8")).hexdigest(), r.image_id


def assign_splits(records: list[PhotoRecord]) -> list[PhotoRecord]:
    """스타일마다 sha1(image_id) 순으로 앞 4장 taste, 다음 10장 test, 나머지 train. 입력 순서와 무관."""
    by_style: dict[str, list[PhotoRecord]] = defaultdict(list)
    for r in records:
        by_style[r.style_id].append(r)
    out: list[PhotoRecord] = []
    for style in sorted(by_style):
        for i, r in enumerate(sorted(by_style[style], key=_order_key)):
            split = "taste" if i < N_TASTE else "test" if i < N_TASTE + N_TEST else "train"
            out.append(replace(r, split=split))
    return out


def split_warnings(records: list[PhotoRecord]) -> list[str]:
    """분할 뒤 train이 10장보다 적은 스타일 경고."""
    n_train = Counter(r.style_id for r in records if r.split == "train")
    return [
        f"경고: 스타일 {s}의 train 사진이 {n_train[s]}장으로 {N_TEST}장보다 적어요"
        for s in sorted({r.style_id for r in records})
        if n_train[s] < N_TEST
    ]
