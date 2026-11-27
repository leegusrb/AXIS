import 'dart:io';
import 'dart:ui';

import 'package:axis_app/local/geometry.dart';
import 'package:axis_app/local/local_analyzer.dart';
import 'package:flutter_test/flutter_test.dart';

Subject person({double cx = 0.5, double dy = 0}) {
  // ai/tests/conftest.make_person과 같은 배치(박스 0.3×0.6, top 0.2)
  const layout = [
    (0.10, 0.0),
    (0.09, -0.08),
    (0.09, 0.08),
    (0.10, -0.16),
    (0.10, 0.16),
    (0.25, -0.30),
    (0.25, 0.30),
    (0.40, -0.40),
    (0.40, 0.40),
    (0.52, -0.42),
    (0.52, 0.42),
    (0.55, -0.15),
    (0.55, 0.15),
    (0.75, -0.15),
    (0.75, 0.15),
    (0.95, -0.15),
    (0.95, 0.15),
  ];
  final kp = [
    for (final (fy, fx) in layout) [cx + fx * 0.3, 0.2 + fy * 0.6, 0.9],
  ];
  kp[6][1] += dy;
  return Subject(
    box: Rect.fromLTWH(cx - 0.15, 0.2, 0.3, 0.6),
    keypoints: kp,
    conf: 0.9,
  );
}

void main() {
  final data = LocalData.fromJsonString(
    File('assets/local/local.json').readAsStringSync(),
  );

  test('local data lists artists and categories', () {
    expect(data.categories.map((c) => c.id), ['person', 'food']);
    final ids = data.artists.map((a) => a.id).toList();
    expect(
      ids,
      containsAll(['wide_full', 'sym_back', 'food_center', 'food_thirds']),
    );
    final b = data.artists.firstWhere((a) => a.id == 'sym_back');
    expect(b.kind, 'rule');
    expect(b.samples, isEmpty);
    expect(b.diagramBox, isNotNull);
    expect(data.foodClasses.length, 80);
  });

  test('learned artist uses stat, rule artist uses rule', () {
    final wide = data.artists.firstWhere((a) => a.id == 'wide_full');
    final r = localResult(
      [person()],
      wide,
      data,
      const {'decode': 1, 'pose': 2},
    );
    expect(r.scorer, 'stat');
    expect(r.modelVersion, '${data.version}-local');
    expect(r.subject!.className, 'person');
    expect(r.timingMs.keys, containsAll(['decode', 'pose', 'guide', 'total']));
    final b = data.artists.firstWhere((a) => a.id == 'sym_back');
    expect(localResult([person()], b, data, const {}).scorer, 'rule');
  });

  test('tilt gives rotate hint; food artist with no food says 음식', () {
    final tight = data.artists.firstWhere((a) => a.id == 'wide_full');
    final r = localResult([person(dy: 0.06)], tight, data, const {});
    expect(r.rotate, 'CW');
    expect(r.rotationDeg, greaterThan(8));
    if (r.direction == 'OK') expect(r.message, '좋아요! 수평만 맞추면 완벽해요');
    final food = data.artists.firstWhere((a) => a.id == 'food_center');
    final n = localResult(const [], food, data, const {});
    expect(n.direction, 'NO_SUBJECT');
    expect(n.message, '음식을 화면에 담아주세요');
    expect(n.category, 'food');
  });
}
