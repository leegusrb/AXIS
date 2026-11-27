// YOLO11 TFLite 출력 해석: (채널, n) 행 우선. 좌표는 입력 크기(size)로 나눈 0–1 값.
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'geometry.dart';

const confThreshold = 0.25;
const iouThreshold = 0.7;
const _maxCandidates = 100;

/// 원본 이미지를 size×size에 넣은 방법: 원본 좌표 = (입력 좌표 − pad) / scale.
class Letterbox {
  const Letterbox({
    required this.scale,
    required this.padX,
    required this.padY,
    required this.srcW,
    required this.srcH,
    required this.size,
  });
  final double scale, padX, padY;
  final int srcW, srcH, size;

  double x(double v) => (v * size - padX) / scale / srcW;
  double y(double v) => (v * size - padY) / scale / srcH;
}

double _iou(Rect a, Rect b) {
  final i = a.intersect(b);
  if (i.width <= 0 || i.height <= 0) return 0;
  final inter = i.width * i.height;
  return inter / (a.width * a.height + b.width * b.height - inter);
}

/// 신뢰도 순으로 겹치는 박스를 지운다(클래스 구분 없음).
List<Subject> _nms(List<Subject> c) {
  c.sort((a, b) => b.conf.compareTo(a.conf));
  final keep = <Subject>[];
  for (final s in c.take(_maxCandidates)) {
    if (keep.every((k) => _iou(k.box, s.box) <= iouThreshold)) keep.add(s);
  }
  return keep;
}

Rect _box(Float32List out, int n, int j, Letterbox lb) {
  final cx = out[j], cy = out[n + j], w = out[2 * n + j], h = out[3 * n + j];
  final x0 = lb.x(cx - w / 2).clamp(0.0, 1.0),
      y0 = lb.y(cy - h / 2).clamp(0.0, 1.0);
  final x1 = lb.x(cx + w / 2).clamp(0.0, 1.0),
      y1 = lb.y(cy + h / 2).clamp(0.0, 1.0);
  return Rect.fromLTRB(x0, y0, math.max(x1, x0), math.max(y1, y0));
}

/// pose 출력 (56, n): cx, cy, w, h, conf, 관절 17×(x, y, conf).
List<Subject> decodePose(Float32List out, int n, Letterbox lb) {
  final cands = <Subject>[];
  for (var j = 0; j < n; j++) {
    final conf = out[4 * n + j];
    if (conf < confThreshold) continue;
    final kp = [
      for (var k = 0; k < 17; k++)
        [
          lb.x(out[(5 + 3 * k) * n + j]).clamp(0.0, 1.0),
          lb.y(out[(6 + 3 * k) * n + j]).clamp(0.0, 1.0),
          out[(7 + 3 * k) * n + j],
        ],
    ];
    cands.add(Subject(box: _box(out, n, j, lb), keypoints: kp, conf: conf));
  }
  return _nms(cands);
}

/// detect 출력 (84, n): cx, cy, w, h, 클래스 80. 서버(ultralytics)처럼 80개 중 1등 클래스를 고르고,
/// 그 클래스가 음식(foodClasses[i] != null)일 때만 남긴다. NMS는 같은 클래스끼리만 한다.
List<Subject> decodeFood(
  Float32List out,
  int n,
  Letterbox lb,
  List<String?> foodClasses,
) {
  final byLabel = <String, List<Subject>>{};
  for (var j = 0; j < n; j++) {
    var best = 0;
    var bestV = out[4 * n + j];
    for (var c = 1; c < foodClasses.length; c++) {
      final v = out[(4 + c) * n + j];
      if (v > bestV) {
        best = c;
        bestV = v;
      }
    }
    final label = foodClasses[best];
    if (bestV < confThreshold || label == null) continue;
    (byLabel[label] ??= []).add(
      Subject(
        box: _box(out, n, j, lb),
        keypoints: List.generate(17, (_) => [0.0, 0.0, 0.0]),
        conf: bestV,
        label: label,
      ),
    );
  }
  return [for (final c in byLabel.values) ..._nms(c)];
}
