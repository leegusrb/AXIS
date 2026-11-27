# 스타일 사진 모으기 안내

AXIS는 "이 스타일 사진처럼 찍으려면 카메라를 어디로 움직일까"를 알려 주는 앱이에요. 그래서 스타일마다 사진이 필요해요. 이 문서는 사진을 모으는 방법을 설명해요.

## 일정

| 언제까지 | 할 일 |
|---|---|
| 11/18 | 스타일 3개 정하기 |
| 11/21 | 스타일당 60장 모으고 manifest 작성하기 |
| 11/27 | 실제 이동 검증 사진 찍기 (아래 3번) |

## 1. 스타일 3개 정하기 (11/18)

- 구도가 뚜렷하게 다른 인물 스타일 3개를 정해요.
- 스타일마다 이름과 "이 스타일에 넣는 기준"을 한두 줄로 적어요.
- 사진 출처 후보도 같이 적어요.
- 지금 `data/styles.json`에는 자리표시용 3개가 들어 있어요: `wide_full`(여백 있는 전신), `tight_close`(꽉 찬 클로즈업), `thirds_upper`(삼분할 상반신). 팀이 11/18에 이름을 바꿔도 돼요.
- 스타일은 한 사진가의 작품이 아니라 여러 사진가의 사진 묶음이에요. 식별자는 `style_id`예요. 사진가 이름을 id로 쓰지 않아요.

## 2. 사진 모으기 (11/21까지)

### 개수와 규칙

- 스타일당 60장이 목표이고, 최소 50장이에요.
- 한 사진가의 사진이 스타일의 1/4(60장이면 15장)을 넘지 않게 해요. 여러 사진가를 섞어야 스타일이 한 사람의 색감에 좌우되지 않아요.
- 인물이 한 명 또렷이 나온 사진만 써요.
- 짧은 변이 400px 이상이어야 해요.
- 아이폰으로 찍은 사진은 설정 > 카메라 > 포맷 > 높은 호환성으로 바꿔 JPEG로 찍어요. 기본(고효율) 포맷은 HEIC라 쓸 수 없어요.

### 쓸 수 있는 라이선스

아래 라이선스이거나, 사진가에게 직접 동의를 받은 사진만 써요.

- `CC0`
- `PDM`
- `CC BY` 계열 (예: `CC BY 4.0`)
- `CC BY-SA` 계열 (예: `CC BY-SA 4.0`)
- `consent` (사진가가 직접 동의함)

그 밖의 라이선스나 라이선스 표기가 없는 사진은 데이터를 만들 때 거부돼요. 쓰지 마세요.

### 찾을 곳

- Wikimedia Commons
- Openverse
- Unsplash, Pexels: 사이트에 적힌 라이선스를 manifest에 그대로 적어요.

### 저장 위치

사진 파일은 `data/raw/<style_id>/` 안에 넣어요. 예: `data/raw/wide_full/001.jpg`. 이 폴더는 git에 올라가지 않아요. 사진을 커밋하지 마세요.

### manifest 작성

`data/manifest.csv`에 사진마다 한 줄씩 적어요. 열은 아래 6개예요.

`image_id,style_id,file,source_url,author,license`

- `file`은 `data/raw/` 기준 상대 경로예요 (`<style_id>/<파일명>`).
- 사진마다 출처, 저작자, 라이선스를 꼭 적어요.
- author나 다른 값에 쉼표가 있으면 값을 큰따옴표로 감싸요. 예: `"Doe, Jane"`

맨 윗줄 헤더와 예시 2줄:

```
image_id,style_id,file,source_url,author,license
wide_full_001,wide_full,wide_full/001.jpg,https://commons.wikimedia.org/wiki/File:Example1.jpg,Jane Doe,CC BY 4.0
tight_close_001,tight_close,tight_close/001.jpg,https://unsplash.com/photos/example2,John Kim,CC0
```

manifest.csv는 git에 커밋해요.

## 3. 실제 이동 검증 사진 (11/27)

앱이 말해 준 방향이 맞는지 확인하는 사진이에요. 아이폰으로 찍어요.

1. 스타일마다 앱이 `OK`를 표시하는 위치에서 기준 사진을 찍어요.
2. 그다음 카메라를 한 방향으로 정해진 만큼 옮겨서 찍어요.
   - 좌우, 상하: 몸을 30cm 평행이동해요.
   - 앞뒤: 한 걸음 이동해요.
3. 6방향(LEFT, RIGHT, UP, DOWN, CLOSER, FARTHER), 스타일 3개, 인물 2명으로 찍어요.
4. 찍을 때마다 파일 이름에 스타일과 방향을 적어요. 확장자는 `.jpg` 또는 `.jpeg`예요(대소문자 무관). HEIC 같은 다른 포맷은 평가에서 건너뛰고 목록으로 알려 줘요.

파일 이름 규칙:

```
<style_id>_<person>_<base|LEFT|RIGHT|UP|DOWN|CLOSER|FARTHER>.jpg
```

- `base`는 `OK` 위치의 기준 사진이에요.
- 나머지는 카메라를 옮긴 방향이에요. 왼쪽으로 옮겨 찍었으면 `LEFT`예요.
- 예: `wide_full_p1_base.jpg`, `wide_full_p1_LEFT.jpg`
- 정답 방향은 옮긴 방향의 반대예요. 왼쪽으로 옮겼으면 정답은 `RIGHT`예요. 이 계산은 코드가 해요. 파일 이름에는 실제로 옮긴 방향을 적으면 돼요.

## 데이터가 다 안 모이면

11/21까지 다 못 모아도 괜찮아요. 모인 만큼으로 먼저 학습해요. 그래도 최소 50장에 가깝게 모아 주세요.
