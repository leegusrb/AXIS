"""스타일 구분이 구도에서 나오는가: 구도 / MobileNet / 둘 다로 test 사진의 스타일을 분류한다.

python -m axis_ai.eval.separability --processed data/processed [--out docs/results/separability.md]
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..data.build import load_embeddings, load_processed
from . import emit, pct


def _accuracy(xtr, ytr, xte, yte) -> float:
    scaler = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(xtr), ytr)
    return float(np.mean(clf.predict(scaler.transform(xte)) == yte))


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="스타일 분류 정확도(구도 vs MobileNet)")
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    emb = load_embeddings(args.processed)
    rows = [r for r in load_processed(args.processed) if r["features"] is not None and r["image_id"] in emb]
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    if not train or not test:
        raise SystemExit("train와 test 사진(인물 검출)이 모두 필요해요")
    styles = sorted({r["style_id"] for r in rows})

    def comp(rs):
        return np.stack([r["features"].vector() for r in rs])

    def mob(rs):
        return np.stack([np.asarray(emb[r["image_id"]], dtype=float) for r in rs])

    def both(rs):
        return np.hstack([comp(rs), mob(rs)])

    ytr = np.array([r["style_id"] for r in train])
    yte = np.array([r["style_id"] for r in test])
    result = {
        "composition": _accuracy(comp(train), ytr, comp(test), yte),
        "mobilenet": _accuracy(mob(train), ytr, mob(test), yte),
        "both": _accuracy(both(train), ytr, both(test), yte),
        "chance": 1 / len(styles),
        "n_train": len(train),
        "n_test": len(test),
        "styles": styles,
    }
    md = "\n".join([
        "# 스타일 구분 가능성",
        "",
        f"스타일 {len(styles)}개, train {len(train)}장, test {len(test)}장 (로지스틱 회귀)",
        "",
        "| 특징 | test 정확도 |",
        "|---|---|",
        f"| 구도 (6차원) | {pct(result['composition'])} |",
        f"| MobileNet (576차원) | {pct(result['mobilenet'])} |",
        f"| 둘 다 | {pct(result['both'])} |",
        f"| 무작위 기준 | {pct(result['chance'])} |",
    ])
    emit(md, args.out)
    return result


if __name__ == "__main__":
    main()
