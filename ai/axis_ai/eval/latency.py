"""추론 시간: inference_logs의 단계별 p50/p95 (깊이·임베딩 열이 있으면 함께, 방식별 전체 시간도).

python -m axis_ai.eval.latency --db server/var/axis.db [--out ...]
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np

from . import emit

STAGES = ("decode_ms", "pose_ms", "guide_ms", "depth_ms", "embed_ms", "total_ms")


def _pct(values) -> dict:
    p50, p95 = (float(v) for v in np.percentile(np.asarray(values, dtype=float), [50, 95]))
    return {"p50": p50, "p95": p95}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="단계별 추론 시간 p50/p95")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(inference_logs)")}
        stages = [s for s in STAGES if s in cols]
        has_scorer = "scorer" in cols
        rows = con.execute(f"SELECT {', '.join(stages)}{', scorer' if has_scorer else ''}"
                           " FROM inference_logs").fetchall()
    finally:
        con.close()

    result: dict = {"count": len(rows)}
    lines = ["# 추론 시간", "", f"요청 {len(rows)}건", ""]
    if not rows:
        lines.append("기록이 없어요.")
        emit("\n".join(lines), args.out)
        return result
    lines += ["| 단계 | p50 (ms) | p95 (ms) |", "|---|---|---|"]
    for i, name in enumerate(stages):
        result[name] = _pct([r[i] for r in rows])
        lines.append(f"| {name[:-3]} | {result[name]['p50']:.1f} | {result[name]['p95']:.1f} |")
    if has_scorer:
        ti = stages.index("total_ms")
        result["by_scorer"] = {}
        lines += ["", "| 방식 | 건수 | 전체 p50 (ms) | 전체 p95 (ms) |", "|---|---|---|---|"]
        for sc in sorted({r[-1] for r in rows}):
            t = [r[ti] for r in rows if r[-1] == sc]
            result["by_scorer"][sc] = {"count": len(t), "total_ms": _pct(t)}
            p = result["by_scorer"][sc]["total_ms"]
            lines.append(f"| {sc} | {len(t)} | {p['p50']:.1f} | {p['p95']:.1f} |")
    emit("\n".join(lines), args.out)
    return result


if __name__ == "__main__":
    main()
