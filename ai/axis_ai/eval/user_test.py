"""사용자 테스트: 가이드 없이 1장 -> 바로 이어서 가이드로 1장 찍은 쌍의 점수 변화.

참가자들이 시연용 폰 한 대를 같이 쓰므로 사람을 구분하지 않는다. 기기마다 사진을 created_at 순으로 보고,
가이드 끔 사진 바로 다음의 가이드 켬 사진을 한 쌍으로 묶는다(사진은 한 쌍에만 쓴다. 끔이 연달아 나오면 가장 나중 것).

python -m axis_ai.eval.user_test --db server/var/axis.db [--since YYYY-MM-DD] [--out ...]
"""
import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import emit, pct


def pair_shots(shots: list[tuple[bool, float]]) -> list[tuple[float, float]]:
    """시간순 (guide_used, score) 목록에서 (끔 점수, 켬 점수) 쌍을 만든다."""
    pairs, pending = [], None
    for used, score in shots:
        if not used:
            pending = score
        elif pending is not None:
            pairs.append((pending, score))
            pending = None
    return pairs


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="가이드 사용 전후 점수 변화")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--since", help="YYYY-MM-DD 이후 사진만")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    cols = {r[1] for r in con.execute("PRAGMA table_info(shots)")}
    who = "COALESCE(CAST(user_id AS TEXT), device_id)" if "user_id" in cols else "device_id"
    sql, params = f"SELECT {who}, guide_used, score FROM shots", []
    if args.since:
        sql += " WHERE created_at >= ?"
        params.append(args.since)
    sql += " ORDER BY created_at, rowid"
    try:
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()

    by_device: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    for device, used, score in rows:
        by_device[device].append((bool(used), float(score)))
    pairs = [p for shots in by_device.values() for p in pair_shots(shots)]

    deltas = [on - off for off, on in pairs]
    result = {
        "n_pairs": len(pairs),
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "improved_share": float(np.mean([d > 0 for d in deltas])) if deltas else None,
        "pairs": [{"off": off, "on": on, "delta": on - off} for off, on in pairs],
    }
    since = f" ({args.since} 이후)" if args.since else ""
    lines = ["# 사용자 테스트", "", f"가이드 없이 1장 -> 바로 이어서 가이드로 1장 찍은 쌍 {len(pairs)}개{since}", ""]
    if deltas:
        lines += [f"- 평균 점수 변화(켬 - 끔): {result['mean_delta']:+.3f}",
                  f"- 점수가 오른 쌍 비율: {pct(result['improved_share'])}", "",
                  "| 쌍 | 끔 | 켬 | 변화 |", "|---|---|---|---|"]
        lines += [f"| {i} | {p['off']:.3f} | {p['on']:.3f} | {p['delta']:+.3f} |"
                  for i, p in enumerate(result["pairs"], 1)]
    else:
        lines.append("비교할 쌍이 없어요.")
    emit("\n".join(lines), args.out)
    return result


if __name__ == "__main__":
    main()
