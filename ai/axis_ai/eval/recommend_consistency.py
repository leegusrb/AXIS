"""추천 일관성: 한 스타일의 taste 사진만 골랐을 때 그 스타일이 1위로 나오는 비율.

python -m axis_ai.eval.recommend_consistency --model models/<v> [--out ...]
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

from ..bundle import ModelBundle
from ..recommend import recommend
from . import emit, pct


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="취향 사진 부분집합에 대한 추천 1위 일치율")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    bundle = ModelBundle.load(args.model)
    style_of = {}
    with open(args.model / "photos.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == "taste":
                style_of[row["image_id"]] = row["style_id"]

    scorers = bundle.scorers["stat"]
    photos: dict[str, list] = {s: [] for s in scorers}
    for t in bundle.taste:
        s = style_of.get(t["image_id"])
        if s in photos:
            photos[s].append(bundle.photo_features(t["image_id"]))

    per_style, hits, total = {}, 0, 0
    for s, items in photos.items():
        h = n = 0
        for k in range(1, len(items) + 1):
            for subset in combinations(items, k):
                n += 1
                h += recommend(list(subset), scorers, bundle.centroids)[0][0] == s
        if n:
            per_style[s] = {"consistency": h / n, "n": n}
        hits, total = hits + h, total + n

    result = {"consistency": hits / total if total else None, "n": total, "per_style": per_style}
    lines = ["# 추천 일관성", "",
             "스타일마다 그 스타일 taste 사진의 모든 부분집합(1장 이상)으로 추천했을 때 1위가 그 스타일인 비율", "",
             "| 스타일 | 일치율 | 조합 수 |", "|---|---|---|"]
    lines += [f"| {s} | {pct(v['consistency'])} | {v['n']} |" for s, v in per_style.items()]
    lines.append(f"| 전체 | {pct(result['consistency']) if total else '-'} | {total} |")
    emit("\n".join(lines), args.out)
    return result


if __name__ == "__main__":
    main()
