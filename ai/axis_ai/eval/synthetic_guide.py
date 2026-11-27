"""안내 방향 정확도(합성): test 원본을 알려진 만큼 어긋나게 만들고 안내가 되돌리는 방향을 가리키는지 본다.

두 가지를 잰다.
- 첫 안내 정확도: 첫 안내 한 번이 변형을 되돌리는 방향인가.
- OK 도달률: 안내를 최대 MAX_STEPS번 따라갔을 때 OK에 닿는가, 몇 단계가 걸리는가.
  안내는 한 번에 가장 효과가 큰 한 가지만 고치므로, 실제 사용에 더 가까운 지표다.

python -m axis_ai.eval.synthetic_guide --processed data/processed --model models/<v> [--scorers stat,rank] [--out ...]
"""
import argparse
from pathlib import Path

from ..bundle import ModelBundle
from ..data.build import load_processed
from ..geometry import transform_person
from ..guide import MOVES, decide
from . import emit, pct

MAGNITUDES = (0.1, 0.2)
MAX_STEPS = 6
DIRECTIONS = tuple(MOVES)


def _sign(v: float) -> int:
    return (v > 0) - (v < 0)


def perturb(person, direction: str, m: float):
    """카메라가 direction의 반대로 움직인 것처럼 인물을 어긋나게 한다. 안내의 정답은 direction."""
    dx, dy, _ = MOVES[direction]
    if direction == "CLOSER":  # 인물이 너무 작다
        return transform_person(person, 0.0, 0.0, 1 / (1 + m))
    if direction == "FARTHER":  # 인물이 너무 크다
        return transform_person(person, 0.0, 0.0, 1 + m)
    return transform_person(person, -m * _sign(dx), -m * _sign(dy), 1.0)


def follow_guidance(person, scorer, max_steps: int = MAX_STEPS) -> int | None:
    """안내를 따라 가상 이동을 반복한다. OK가 될 때까지 걸린 단계 수, 닿지 못하면 None."""
    for step in range(max_steps + 1):
        d = decide(person, scorer).direction
        if d == "OK":
            return step
        if d not in MOVES or step == max_steps:
            return None
        nxt = transform_person(person, *MOVES[d])
        if nxt is None:
            return None
        person = nxt
    return None


def evaluate(rows: list[dict], bundle: ModelBundle, kinds: list[str]) -> dict:
    out = {}
    for kind in kinds:
        per_dir = {d: [0, 0] for d in DIRECTIONS}  # 맞음, 전체
        rec_dir = {d: 0 for d in DIRECTIONS}  # OK 도달
        steps: list[int] = []
        ok_count = total = 0
        for r in rows:
            scorer = bundle.scorers.get(kind, {}).get(r["style_id"])
            if scorer is None:
                continue
            for d in DIRECTIONS:
                for m in MAGNITUDES:
                    variant = perturb(r["person"], d, m)
                    if variant is None:
                        continue
                    got = decide(variant, scorer).direction
                    total += 1
                    ok_count += got == "OK"
                    per_dir[d][1] += 1
                    per_dir[d][0] += got == d
                    k = follow_guidance(variant, scorer)
                    if k is not None:
                        rec_dir[d] += 1
                        steps.append(k)
        if total == 0:
            continue
        out[kind] = {
            "accuracy": sum(c for c, _ in per_dir.values()) / total,
            "n": total,
            "ok_share": ok_count / total,
            "per_direction": {d: (c / n if n else None) for d, (c, n) in per_dir.items()},
            "recovery": len(steps) / total,
            "mean_steps": (sum(steps) / len(steps)) if steps else None,
            "recovery_per_direction": {d: (rec_dir[d] / n if n else None) for d, (_, n) in per_dir.items()},
        }
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="합성 이동에 대한 안내 방향 정확도")
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--scorers", default="stat,rank")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    bundle = ModelBundle.load(args.model)
    rows = [r for r in load_processed(args.processed) if r["split"] == "test" and r["person"] is not None]
    kinds = [k.strip() for k in args.scorers.split(",") if k.strip()]
    result = evaluate(rows, bundle, kinds)

    lines = ["# 안내 방향 정확도 (합성)", "",
             f"test 사진 {len(rows)}장, 방향 6개 x 크기 {MAGNITUDES}. 정답은 변형을 되돌리는 방향(이동 방향의 반대).", "",
             "| Scorer | 전체 | " + " | ".join(DIRECTIONS) + " | OK 응답 비율 | 건수 |",
             "|---|---|" + "---|" * len(DIRECTIONS) + "---|---|"]
    for kind, v in result.items():
        cells = " | ".join("-" if v["per_direction"][d] is None else pct(v["per_direction"][d]) for d in DIRECTIONS)
        lines.append(f"| {kind} | {pct(v['accuracy'])} | {cells} | {pct(v['ok_share'])} | {v['n']} |")
    if not result:
        lines.append("| (평가할 Scorer 없음) |")
    lines += ["", f"## 안내를 따라갔을 때 OK 도달 (최대 {MAX_STEPS}단계)", "",
              "| Scorer | OK 도달 | " + " | ".join(DIRECTIONS) + " | 평균 단계 |",
              "|---|---|" + "---|" * len(DIRECTIONS) + "---|"]
    for kind, v in result.items():
        cells = " | ".join("-" if v["recovery_per_direction"][d] is None else pct(v["recovery_per_direction"][d]) for d in DIRECTIONS)
        ms = "-" if v["mean_steps"] is None else f"{v['mean_steps']:.1f}"
        lines.append(f"| {kind} | {pct(v['recovery'])} | {cells} | {ms} |")
    emit("\n".join(lines), args.out)
    return result


if __name__ == "__main__":
    main()
