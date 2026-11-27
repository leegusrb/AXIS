# AXIS

인물 사진을 찍을 때 원하는 스타일의 구도로 카메라를 어느 쪽으로 움직일지 알려 주는 AI 구도 안내 앱이다. Python AI, FastAPI 서버, Flutter 앱으로 이루어진다.

## 폴더 구조

```
ai/        Python 패키지 axis_ai (특징 추출, Scorer, 안내, 추천, 데이터 도구, 평가)
server/    FastAPI 앱
app/       Flutter 앱
data/      로컬 데이터. manifest.csv와 styles.json만 커밋
docs/      평가 결과, 팀원 작업 안내
```

## 설치

저장소 루트에서 한 번만 실행한다(Python 3.12).

```
python3 -m venv .venv
.venv/bin/pip install -e "ai[dev]" -e "server[dev]"
cd app && flutter pub get
```

## 데이터 만들기와 학습

사진은 `data/raw/<style_id>/`에 두고 `data/manifest.csv`에 출처를 적는다(`docs/team/data-collection.md`). 저장소 루트에서:

```
.venv/bin/python -m axis_ai.data.build --manifest data/manifest.csv --raw data/raw --out data/processed
.venv/bin/python -m axis_ai.train --processed data/processed --styles data/styles.json --out models/v1 [--rank]
```

- `--rank`를 주면 stat scorer와 함께 순위 학습 scorer(`rank`)도 만든다.
- 같은 `--out`으로 다시 학습하면 이전 `stat/`, `rank/` 폴더를 지우고 새로 쓴다.
- 결과 폴더 이름(`v1`)이 모델 버전이다.

## 모델 모드로 서버 켜기

학습한 번들을 서버에 연결하려면 `AXIS_MODE=model`(기본값)과 `AXIS_MODELS_DIR`를 준다. 기본 scorer는 `AXIS_DEFAULT_SCORER`(기본 `stat`)로 바꾼다. `rank`를 기본으로 쓰려면 `--rank`로 학습해 두어야 하고, 쓸 수 없는 값이면 서버가 시작할 때 알려 주고 종료한다.

```
AXIS_MODE=model AXIS_MODELS_DIR=models/v1 AXIS_DEFAULT_SCORER=stat .venv/bin/python -m axis_server --port 8000
```

## 서버 실행

저장소 루트에서 `AXIS_MODE`를 골라 실행한다. 기본 포트는 8000이고 `--host`, `--port`로 바꾼다.

```
AXIS_MODE=fake .venv/bin/python -m axis_server --port 8000
AXIS_MODE=rule .venv/bin/python -m axis_server --port 8000
AXIS_MODE=model AXIS_MODELS_DIR=models/<version> .venv/bin/python -m axis_server --port 8000
```

- `fake`: 가짜 카탈로그(단색 이미지)와 가짜 파이프라인. 모델과 사진 없이 앱 화면을 개발할 때 쓴다.
- `rule`: 가짜 카탈로그와 실제 YOLO 포즈 검출, 규칙 기반 scorer(`rule`). 사진 없이 실기기에서 인물 인식과 안내 방향을 확인할 때 쓴다. 처음 실행하면 `yolo11n-pose` 가중치를 내려받는다.
- `model`: `AXIS_MODELS_DIR`의 학습된 모델 번들. 사진은 `AXIS_RAW_DIR`(기본 `data/raw`)에서 제공한다.

폰에서 접속하려면 노트북과 폰이 같은 Wi-Fi에 있어야 한다. 노트북 IP는 `ipconfig getifaddr en0`로 찾고, 앱의 서버 주소를 `http://<IP>:8000`으로 설정한다.

## 앱 실행

1. 서버를 켠다(위 "서버 실행").
2. 아이폰을 연결하고 `app/`에서 `flutter run`을 실행한다. Xcode에서 Runner의 Signing에 Apple ID 팀을 지정해야 한다.
3. 앱의 설정 화면에서 서버 주소(`http://<노트북 IP>:8000`)를 입력한다.

실기기 점검 항목은 `docs/results/device-check.md`에 있다.

## 평가

사용자 테스트(`user_test`)는 참가자들이 시연용 폰 한 대를 같이 쓰는 것을 전제로 한다. "가이드 없이 1장 -> 바로 이어서 가이드로 1장" 순서로 찍게 하면, 기기별로 시간순에서 끔 사진과 그 다음 켬 사진을 한 쌍으로 묶어 쌍의 개수, 평균 점수 변화, 점수가 오른 쌍 비율을 낸다.

발표용 수치를 만드는 스크립트. 저장소 루트에서 실행하고 `--out`을 주면 Markdown을 파일로도 쓴다(결과는 `docs/results/`에 모은다).

```
.venv/bin/python -m axis_ai.eval.separability --processed data/processed --out docs/results/separability.md
.venv/bin/python -m axis_ai.eval.synthetic_guide --processed data/processed --model models/<version> --out docs/results/synthetic-guide.md
.venv/bin/python -m axis_ai.eval.recommend_consistency --model models/<version> --out docs/results/recommend.md
.venv/bin/python -m axis_ai.eval.real_moves --dir <실제 이동 사진 폴더> --model models/<version> [--style-filter <style_id>] --out docs/results/real-moves.md
.venv/bin/python -m axis_ai.eval.latency --db server/var/axis.db --out docs/results/latency.md
.venv/bin/python -m axis_ai.eval.user_test --db server/var/axis.db [--since YYYY-MM-DD] --out docs/results/user-test.md
```

## 데이터 빌드

python.org에서 설치한 Python에서는 처음 모델을 내려받을 때 `CERTIFICATE_VERIFY_FAILED`가 날 수 있다. 저장소 루트에서 `export SSL_CERT_FILE=$(.venv/bin/python -m certifi)`를 실행한 뒤 다시 돌리면 된다.
