"""발표용 평가 스크립트. 각 모듈은 `python -m axis_ai.eval.<이름>`으로 실행한다."""
from pathlib import Path

# 카메라 이동 방향의 반대. 인물을 한쪽으로 옮긴 변형의 정답은 그 변형을 되돌리는 카메라 이동이다.
OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP",
            "CLOSER": "FARTHER", "FARTHER": "CLOSER"}


def emit(md: str, out: Path | None) -> None:
    """Markdown을 표준출력에 찍고, out이 있으면 파일로도 쓴다."""
    print(md)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"
