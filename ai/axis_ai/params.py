"""실기기에서 조정할 수 있도록 모아 둔 상수."""

MOVE_STEP = 0.08          # 이동 후보 크기(정규화)
ZOOM_IN = 1.25            # 확대 후보 배율
ZOOM_OUT = 0.8            # 축소 후보 배율
OK_DELTA = 0.03           # OK 판정 점수 상승폭 기준
OK_SCORE = 0.6            # OK 판정 점수 기준
LARGE_LOG_GAIN = 0.5      # LARGE 판정 기준(로그 점수 상승폭)
MIN_PERSON_CONF = 0.4     # 주인공 최소 신뢰도
REC_W_COMPOSITION = 0.6   # 추천 가중치(구도)
REC_W_EMBED = 0.4         # 추천 가중치(임베딩)
REC_TEMPERATURE = 0.1     # softmax 온도
MIN_VISIBLE_AREA = 0.3    # 화면에 남아야 하는 박스 면적 비율
KEYPOINT_VISIBLE = 0.3    # 키포인트가 "보인다"고 보는 최소 신뢰도
SHOT_MIN_STD = 0.25       # shot 차원 표준편차 하한(한 가지 샷만 있는 스타일에서 점수가 0으로 붕괴하는 것을 막음)
ROTATE_DEG = 8.0          # 회전(수평) 안내를 띄우는 기울기(도)
DEPTH_SIZE = 256          # 깊이 추정 입력 짧은 변(px)
DEPTH_PREVIEW_W = 128     # 깊이 보기 미리보기 너비(px)
HEAD_OK = 0.5             # Guide Head 보정값이 이동 한 단계의 이 비율보다 작으면 OK
ANGLE_NORM = 30.0         # Guide Head 회전 출력 정규화(도)
