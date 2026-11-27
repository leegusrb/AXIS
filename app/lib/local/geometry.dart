// ai/axis_ai/features.py, geometry.py, rotation.py를 옮긴 것. 값은 test/local_parity_test.dart가 파이썬과 맞춘다.
import 'dart:math' as math;
import 'dart:ui' show Rect;

class LocalParams {
  LocalParams(Map<String, dynamic> j)
    : moveStep = _d(j['MOVE_STEP']),
      zoomIn = _d(j['ZOOM_IN']),
      zoomOut = _d(j['ZOOM_OUT']),
      okDelta = _d(j['OK_DELTA']),
      okScore = _d(j['OK_SCORE']),
      largeLogGain = _d(j['LARGE_LOG_GAIN']),
      minPersonConf = _d(j['MIN_PERSON_CONF']),
      minVisibleArea = _d(j['MIN_VISIBLE_AREA']),
      keypointVisible = _d(j['KEYPOINT_VISIBLE']),
      rotateDeg = _d(j['ROTATE_DEG']);

  factory LocalParams.fromJson(Map<String, dynamic> j) => LocalParams(j);

  final double moveStep, zoomIn, zoomOut, okDelta, okScore, largeLogGain;
  final double minPersonConf, minVisibleArea, keypointVisible, rotateDeg;
}

double _d(Object? v) => (v as num).toDouble();

/// 검출된 피사체. box는 0–1 정규화(left, top, width, height), keypoints는 17×(x, y, conf).
class Subject {
  const Subject({
    required this.box,
    required this.keypoints,
    required this.conf,
    this.label = 'person',
  });
  final Rect box;
  final List<List<double>> keypoints;
  final double conf;
  final String label;
}

class Features {
  const Features(
    this.cx,
    this.eyeY,
    this.area,
    this.headroom,
    this.faceX,
    this.shot,
  );
  final double cx, eyeY, area, headroom, faceX, shot;

  /// 파이썬 특징 이름으로 값 꺼내기(규칙 점수·테스트용).
  double get(String name) => switch (name) {
    'cx' => cx,
    'eye_y' => eyeY,
    'area' => area,
    'headroom' => headroom,
    'face_x' => faceX,
    'shot' => shot,
    _ => throw ArgumentError(name),
  };

  List<double> get vector => [cx, eyeY, area, headroom, faceX, shot];
}

const _nose = 0;
const _eyes = [1, 2];
const _knees = [13, 14];
const _ankles = [15, 16];

Subject? selectSubject(List<Subject> all, LocalParams p) {
  Subject? best;
  var bestV = -1.0;
  for (final s in all) {
    if (s.conf < p.minPersonConf) continue;
    final v = s.box.width * s.box.height * s.conf;
    if (v > bestV) {
      best = s;
      bestV = v;
    }
  }
  return best;
}

List<int> _visible(Subject s, List<int> idx, LocalParams p) => [
  for (final i in idx)
    if (s.keypoints[i][2] >= p.keypointVisible) i,
];

Features computeFeatures(Subject s, LocalParams p) {
  final b = s.box;
  final cx = b.left + b.width / 2;
  final kp = s.keypoints;
  final eyes = _visible(s, _eyes, p);
  final nose = _visible(s, [_nose], p);
  double mean(List<int> idx, int c) =>
      idx.map((i) => kp[i][c]).reduce((a, b) => a + b) / idx.length;
  final eyeY = eyes.isNotEmpty
      ? mean(eyes, 1)
      : nose.isNotEmpty
      ? kp[_nose][1]
      : b.top + 0.1 * b.height;
  final face = _visible(s, [_nose, ..._eyes], p);
  final faceX = face.isNotEmpty ? mean(face, 0) : cx;
  final shot = _visible(s, _ankles, p).isNotEmpty
      ? 1.0
      : _visible(s, _knees, p).isNotEmpty
      ? 0.5
      : 0.0;
  return Features(cx, eyeY, b.width * b.height, b.top, faceX, shot);
}

/// 박스 중심 기준 scale배 후 (dx, dy) 이동. 화면에 남는 면적이 너무 작으면 null.
Subject? transformSubject(
  Subject s,
  double dx,
  double dy,
  double scale,
  LocalParams p,
) {
  final b = s.box;
  final cx = b.left + b.width / 2, cy = b.top + b.height / 2;
  final ncx = cx + dx, ncy = cy + dy;
  final nw = b.width * scale, nh = b.height * scale;
  final x0 = ncx - nw / 2,
      y0 = ncy - nh / 2,
      x1 = ncx + nw / 2,
      y1 = ncy + nh / 2;
  final cx0 = math.max(x0, 0.0), cy0 = math.max(y0, 0.0);
  final cx1 = math.min(x1, 1.0), cy1 = math.min(y1, 1.0);
  final cw = math.max(cx1 - cx0, 0.0), ch = math.max(cy1 - cy0, 0.0);
  final full = nw * nh;
  if (full <= 0 || (cw * ch) / full < p.minVisibleArea) return null;
  final kp = [
    for (final k in s.keypoints)
      () {
        final x = (k[0] - cx) * scale + ncx, y = (k[1] - cy) * scale + ncy;
        final out = x < 0 || x > 1 || y < 0 || y > 1;
        return [x, y, out ? 0.0 : k[2]];
      }(),
  ];
  return Subject(
    box: Rect.fromLTWH(cx0, cy0, cw, ch),
    keypoints: kp,
    conf: s.conf,
    label: s.label,
  );
}

const _tiltPairs = [
  ([5, 6], 0.25),
  ([1, 2], 0.10),
];

/// 어깨(없으면 눈) 선의 기울기(도). 양수면 화면 오른쪽이 아래.
double? estimateTilt(Subject s, LocalParams p) {
  if (s.label != 'person') return null;
  final kp = s.keypoints;
  for (final (pair, minGap) in _tiltPairs) {
    final a = kp[pair[0]], b = kp[pair[1]];
    if (a[2] < p.keypointVisible || b[2] < p.keypointVisible) continue;
    final (l, r) = (a[0] < b[0] || (a[0] == b[0] && a[1] <= b[1]))
        ? (a, b)
        : (b, a);
    if (r[0] - l[0] < minGap * s.box.width) return null;
    return math.atan2(r[1] - l[1], r[0] - l[0]) * 180 / math.pi;
  }
  return null;
}

String? rotateHint(double? deg, LocalParams p) {
  if (deg == null || deg.abs() <= p.rotateDeg) return null;
  return deg > 0 ? 'CW' : 'CCW';
}
