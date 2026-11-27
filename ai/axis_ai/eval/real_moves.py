"""안내 방향 정확도(실제): 실제로 카메라를 옮겨 찍은 사진에서 안내가 이동의 반대인지 본다.

파일 이름: <style_id>_<person>_<base|LEFT|RIGHT|UP|DOWN|CLOSER|FARTHER>.jpg (style_id에 밑줄이 있을 수 있어 오른쪽부터 읽는다)

python -m axis_ai.eval.real_moves --dir data/real_moves --model models/<v> [--style-filter <id>] [--out ...]
"""
import argparse
from pathlib import Path

from ..bundle import ModelBundle
from ..data.manifest import open_image
from ..features import select_subject
from ..guide import decide
from . import OPPOSITE, emit, pct


_JPEG_SUFFIXES = (".jpg", ".jpeg")


def parse_name(path: Path) -> tuple[str, str, str] | None:
    """(style_id, person, move). 규칙에 맞지 않으면 None."""
    parts = path.stem.rsplit("_", 2)
    if len(parts) != 3 or not parts[0] or not parts[1]:
        return None
    style_id, person, move = parts
    if move != "base" and move not in OPPOSITE:
        return None
    return style_id, person, move


def run(folder: Path, bundle: ModelBundle, detector, kinds: list[str], style_filter: str | None = None) -> dict:
    stats = {k: {"correct": 0, "n": 0, "misses": []} for k in kinds}
    no_person, no_scorer, skipped = [], [], []
    for path in sorted(p for p in Path(folder).iterdir() if p.is_file()):
        parsed = parse_name(path)
        if parsed is not None and path.suffix.lower() not in _JPEG_SUFFIXES:
            skipped.append(path.name)  # 예: 아이폰 기본 포맷 .HEIC
            continue
        if parsed is None or parsed[2] == "base":
            continue
        style_id, _, move = parsed
        if style_filter and style_id != style_filter:
            continue
        subject = select_subject(detector.detect(open_image(path)))
        if subject is None:
            no_person.append(path.name)
            continue
        expected = OPPOSITE[move]
        used = False
        for kind in kinds:
            scorer = bundle.scorers.get(kind, {}).get(style_id)
            if scorer is None:
                continue
            used = True
            got = decide(subject, scorer).direction
            stats[kind]["n"] += 1
            if got == expected:
                stats[kind]["correct"] += 1
            else:
                stats[kind]["misses"].append({"file": path.name, "expected": expected, "got": got})
        if not used:
            no_scorer.append(path.name)
    scorers = {
        k: {"accuracy": (v["correct"] / v["n"]) if v["n"] else None, "n": v["n"], "misses": v["misses"]}
        for k, v in stats.items()
    }
    return {"scorers": scorers, "no_person": no_person, "no_scorer": no_scorer, "skipped_non_jpeg": skipped}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="실제 이동 사진에 대한 안내 방향 정확도")
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--style-filter")
    ap.add_argument("--scorers", default="stat,rank")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    from ..pose import PoseDetector

    bundle = ModelBundle.load(args.model)
    kinds = [k.strip() for k in args.scorers.split(",") if k.strip()]
    result = run(args.dir, bundle, PoseDetector(), kinds, args.style_filter)

    lines = ["# 안내 방향 정확도 (실제 이동)", "", "| Scorer | 정확도 | 사진 수 |", "|---|---|---|"]
    for kind, v in result["scorers"].items():
        lines.append(f"| {kind} | {'-' if v['accuracy'] is None else pct(v['accuracy'])} | {v['n']} |")
    lines += ["", f"인물을 찾지 못한 사진: {len(result['no_person'])}장"]
    lines += [f"- {n}" for n in result["no_person"]]
    if result["skipped_non_jpeg"]:
        lines += ["", f"JPEG가 아니라 건너뛴 파일: {len(result['skipped_non_jpeg'])}개 (아이폰은 설정 > 카메라 > 포맷 > 높은 호환성)"]
        lines += [f"- {n}" for n in result["skipped_non_jpeg"]]
    for kind, v in result["scorers"].items():
        if v["misses"]:
            lines += ["", f"## {kind} 틀린 사진", ""]
            lines += [f"- {m['file']}: 정답 {m['expected']}, 안내 {m['got']}" for m in v["misses"]]
    emit("\n".join(lines), args.out)
    return result


if __name__ == "__main__":
    main()
