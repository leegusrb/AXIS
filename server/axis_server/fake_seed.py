"""사진 없이 앱을 개발할 수 있는 가짜 카탈로그. 단색 JPEG를 만들어 둔다."""
import math
from pathlib import Path

from PIL import Image

_STYLES = [
    ("wide_full", "여백 있는 전신", "인물 주변에 여백을 두고 전신을 담아요", (120, 160, 200)),
    ("tight_close", "꽉 찬 클로즈업", "얼굴과 상반신으로 화면을 가득 채워요", (200, 130, 120)),
    ("thirds_upper", "삼분할 상반신", "상반신을 화면 왼쪽 삼분할 선에 맞춰요", (130, 190, 140)),
]
_SIZE = (480, 640)
_DIAGRAM = {"wide_full": (0.4, 0.15, 0.2, 0.7), "tight_close": (0.2, 0.15, 0.6, 0.85),
            "thirds_upper": (0.2, 0.2, 0.3, 0.8)}


def _write(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", _SIZE, color).save(path, "JPEG", quality=85)


class FakeSeedCatalog:
    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self._styles = []
        self._taste = []
        self._taste_style: dict[str, str] = {}
        for sid, name, desc, color in _STYLES:
            samples = []
            for k in range(3):
                f = f"{sid}_{k + 1}.jpg"
                _write(self.raw_dir / f, tuple(min(255, c + 20 * k) for c in color))
                samples.append({"file": f, "author": "AXIS 가짜 데이터", "license": "CC0"})
            self._styles.append({"id": sid, "name": name, "description": desc, "samples": samples,
                                 "diagram_box": _DIAGRAM[sid]})
        for n in range(12):
            sid, color = _STYLES[n % 3][0], _STYLES[n % 3][3]
            iid = f"taste_{n + 1:02d}"
            _write(self.raw_dir / f"{iid}.jpg", tuple(max(0, c - 5 * (n // 3)) for c in color))
            self._taste.append({"image_id": iid, "file": f"{iid}.jpg"})
            self._taste_style[iid] = sid

    def styles(self) -> list[dict]:
        return self._styles

    def taste_photos(self) -> list[dict]:
        return self._taste

    def recommend(self, image_ids: list[str]) -> list[tuple[str, float]]:
        counts = {s[0]: 0 for s in _STYLES}
        for i in image_ids:
            counts[self._taste_style[i]] += 1  # 모르는 id는 KeyError
        exps = {s: math.exp(1.5 * c) for s, c in counts.items()}
        total = sum(exps.values())
        return sorted(((s, e / total * 100) for s, e in exps.items()), key=lambda t: (-t[1], t[0]))


def make_fake_catalog(dir: Path) -> FakeSeedCatalog:
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    return FakeSeedCatalog(d)
