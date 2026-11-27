"""폰 분석용 자산 내보내기: 학습값·규칙·작가 목록(local.json), TFLite 모델, Dart 테스트용 정답값.

python -m axis_ai.export_app --model models/v4 --tflite models/v4/export --app app
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import params
from .artists import ARTIST_INFO, CATEGORIES, RULE_ARTISTS, RULE_DIAGRAM, learned_diagram
from .bundle import ModelBundle
from .detect import FOOD_CLASSES
from .features import compute_features
from .guide import decide
from .rotation import estimate_tilt
from .scorers.rule import AnyOfScorer
from .types import FEATURE_NAMES, Person

_PARAMS = ("MOVE_STEP", "ZOOM_IN", "ZOOM_OUT", "OK_DELTA", "OK_SCORE", "LARGE_LOG_GAIN", "MIN_PERSON_CONF",
           "MIN_VISIBLE_AREA", "KEYPOINT_VISIBLE", "ROTATE_DEG")
# (박스 높이 기준 y 비율, 박스 너비 기준 x 오프셋 비율), COCO 17점 — 테스트용 인물 배치
_LAYOUT = [(0.10, 0.0), (0.09, -0.08), (0.09, 0.08), (0.10, -0.16), (0.10, 0.16), (0.25, -0.30), (0.25, 0.30),
           (0.40, -0.40), (0.40, 0.40), (0.52, -0.42), (0.52, 0.42), (0.55, -0.15), (0.55, 0.15),
           (0.75, -0.15), (0.75, 0.15), (0.95, -0.15), (0.95, 0.15)]
_VIS = {"all": range(17), "upper": range(11), "face": range(5)}
LETTERBOX = 320


def _rule_json(sc) -> list[dict]:
    parts = sc.parts if isinstance(sc, AnyOfScorer) else [sc]
    return [{"target": p.target, "spread": p.spread} for p in parts]


def build_local_json(bundle: ModelBundle, food_names: list[str]) -> dict:
    artists = []
    for s in bundle.styles:
        info = ARTIST_INFO.get(s["id"], {})
        artists.append({"id": s["id"], "name": info.get("name", s["name"]),
                        "description": info.get("description", s["description"]),
                        "category": info.get("category", "person"), "kind": "learned",
                        "diagram_box": list(learned_diagram(bundle.rows, s["id"]) or (0.35, 0.2, 0.3, 0.6))})
    for sid in RULE_ARTISTS:
        info = ARTIST_INFO[sid]
        artists.append({"id": sid, "name": info["name"], "description": info["description"],
                        "category": info["category"], "kind": "rule", "diagram_box": list(RULE_DIAGRAM[sid])})
    return {
        "version": bundle.version,
        "params": {k: getattr(params, k) for k in _PARAMS},
        "stat": {sid: {"means": sc.means.tolist(), "inv_covs": sc.inv_covs.tolist()}
                 for sid, sc in bundle.scorers["stat"].items()},
        "rules": {sid: _rule_json(sc) for sid, sc in RULE_ARTISTS.items()},
        "artists": artists,
        "categories": CATEGORIES,
        "food_classes": [n if n in FOOD_CLASSES else None for n in food_names],
    }


def _person(rng: np.random.Generator) -> Person:
    w, h = rng.uniform(0.08, 0.6), rng.uniform(0.2, 0.95)
    cx, top = rng.uniform(0.1, 0.9), rng.uniform(-0.1, 0.5)
    vis = set(_VIS[rng.choice(list(_VIS))])
    kp = np.zeros((17, 3))
    for i, (fy, fx) in enumerate(_LAYOUT):
        kp[i] = (cx + fx * w, top + fy * h, rng.uniform(0.35, 0.99) if i in vis else rng.uniform(0, 0.29))
    kp[6, 1] += rng.normal(0, 0.03)  # 어깨 기울기
    return Person(box=(cx - w / 2, top, w, h), keypoints=kp, conf=float(rng.uniform(0.4, 0.99)))


def _decision(d) -> dict:
    return {"direction": d.direction, "magnitude": d.magnitude, "message": d.message,
            "score": d.score, "score_delta": d.score_delta,
            "target_box": list(d.target_box) if d.target_box else None}


def _person_json(p: Person) -> dict:
    return {"box": list(p.box), "keypoints": p.keypoints.tolist(), "conf": p.conf, "label": p.label}


def make_parity(bundle: ModelBundle, n_person: int = 40, n_food: int = 5, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    cases = []
    people_scorers = {**bundle.scorers["stat"], "sym_back": RULE_ARTISTS["sym_back"]}
    for _ in range(n_person):
        p = _person(rng)
        f = compute_features(p)
        cases.append({"_person": p, "person": _person_json(p),
                      "features": {k: float(getattr(f, k)) for k in FEATURE_NAMES},
                      "scores": {s: float(sc.score(f)) for s, sc in people_scorers.items()},
                      "decide": {s: _decision(decide(p, sc)) for s, sc in people_scorers.items()},
                      "tilt": estimate_tilt(p)})
    for _ in range(n_food):
        w = rng.uniform(0.1, 0.7)
        x, y = rng.uniform(0, 1 - w), rng.uniform(0, 1 - w)
        p = Person(box=(x, y, w, w), keypoints=np.zeros((17, 3)), conf=float(rng.uniform(0.4, 0.99)), label="cake")
        f = compute_features(p)
        food = {s: RULE_ARTISTS[s] for s in ("food_center", "food_thirds")}
        cases.append({"_person": p, "person": _person_json(p),
                      "features": {k: float(getattr(f, k)) for k in FEATURE_NAMES},
                      "scores": {s: float(sc.score(f)) for s, sc in food.items()},
                      "decide": {s: _decision(decide(p, sc, "음식")) for s, sc in food.items()},
                      "tilt": None})
    return cases


def pose_reference(tflite: Path, image: Path) -> tuple[np.ndarray, dict]:
    """사진 한 장의 pose 출력과 letterbox 값, 가장 신뢰도가 높은 사람 박스(Dart 해석 테스트용)."""
    from ai_edge_litert.interpreter import Interpreter
    from PIL import Image

    img = Image.open(image).convert("RGB")
    W, H, S = img.width, img.height, LETTERBOX
    s = min(S / W, S / H)
    nw, nh = round(W * s), round(H * s)
    px, py = (S - nw) // 2, (S - nh) // 2
    canvas = Image.new("RGB", (S, S), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    x = (np.asarray(canvas, np.float32) / 255).transpose(2, 0, 1)[None]
    it = Interpreter(model_path=str(tflite))
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], x)
    it.invoke()
    out = it.get_tensor(it.get_output_details()[0]["index"])[0]  # (56, N)
    j = int(out[4].argmax())
    cx, cy, w, h = (float(v) * S for v in out[:4, j])
    box = [(cx - w / 2 - px) / s / W, (cy - h / 2 - py) / s / H, w / s / W, h / s / H]
    lb = {"scale": s, "pad_x": px, "pad_y": py, "src_w": W, "src_h": H}
    return out.astype(np.float32), {"letterbox": lb, "n": int(out.shape[1]), "top_box": box,
                                    "top_conf": float(out[4, j])}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="폰 분석용 자산 내보내기")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tflite", type=Path, required=True)
    ap.add_argument("--app", type=Path, default=Path("app"))
    ap.add_argument("--frame", type=Path, default=Path("data/raw/wide_full/ov_05c14e4a.jpg"))
    args = ap.parse_args(argv)
    from ultralytics import YOLO

    from .detect import DEFAULT_FOOD_WEIGHTS

    bundle = ModelBundle.load(args.model)
    names = YOLO(str(DEFAULT_FOOD_WEIGHTS)).names
    assets = args.app / "assets" / "local"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "local.json").write_text(
        json.dumps(build_local_json(bundle, [names[i] for i in range(len(names))]), ensure_ascii=False),
        encoding="utf-8")
    for f in ("yolo11n-pose.tflite", "yolo11n.tflite"):
        shutil.copyfile(args.tflite / f, assets / f)
    fixtures = args.app / "test" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    cases = [{k: v for k, v in c.items() if k != "_person"} for c in make_parity(bundle)]
    out, pose = pose_reference(args.tflite / "yolo11n-pose.tflite", args.frame)
    out.tofile(fixtures / "pose_out.bin")  # float32 리틀엔디언, (56, N) 행 우선
    (fixtures / "local_parity.json").write_text(json.dumps({"cases": cases, "pose_frame": pose},
                                                           ensure_ascii=False), encoding="utf-8")
    print(f"자산 -> {assets}, 정답값 {len(cases)}개 -> {fixtures}")


if __name__ == "__main__":
    main()
