import 'dart:convert';
import 'dart:io';
import 'dart:ui';

import 'package:axis_app/local/geometry.dart';
import 'package:axis_app/local/guide.dart';
import 'package:axis_app/local/scorers.dart';
import 'package:flutter_test/flutter_test.dart';

Map<String, dynamic> _json(String path) =>
    jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;

Subject subjectFromJson(Map<String, dynamic> j) {
  final b = (j['box'] as List).map((e) => (e as num).toDouble()).toList();
  return Subject(
    box: Rect.fromLTWH(b[0], b[1], b[2], b[3]),
    keypoints: [
      for (final r in j['keypoints'] as List)
        [for (final v in r as List) (v as num).toDouble()],
    ],
    conf: (j['conf'] as num).toDouble(),
    label: j['label'] as String,
  );
}

void main() {
  final local = _json('assets/local/local.json');
  final params = LocalParams.fromJson(local['params'] as Map<String, dynamic>);
  final scorers = loadScorers(local);
  final cases = (_json('test/fixtures/local_parity.json')['cases'] as List)
      .cast<Map<String, dynamic>>();

  test('scorers cover all artists', () {
    expect(scorers.keys.toSet(), {
      'wide_full',
      'tight_close',
      'thirds_upper',
      'sym_back',
      'food_center',
      'food_thirds',
    });
  });

  test('features, scores, decisions and tilt match python', () {
    expect(cases.length, 45);
    for (final c in cases) {
      final p = subjectFromJson(c['person'] as Map<String, dynamic>);
      final f = computeFeatures(p, params);
      final want = c['features'] as Map<String, dynamic>;
      for (final k in want.keys) {
        expect(f.get(k), closeTo((want[k] as num).toDouble(), 1e-9), reason: k);
      }
      final noun = p.label == 'person' ? '인물' : '음식';
      for (final e in (c['scores'] as Map<String, dynamic>).entries) {
        expect(
          scorers[e.key]!.score(f),
          closeTo((e.value as num).toDouble(), 1e-9),
          reason: e.key,
        );
      }
      for (final e in (c['decide'] as Map<String, dynamic>).entries) {
        final d = decide(p, scorers[e.key]!, params, noun: noun);
        final w = e.value as Map<String, dynamic>;
        expect(d.direction, w['direction'], reason: e.key);
        expect(d.magnitude, w['magnitude'], reason: e.key);
        expect(d.message, w['message'], reason: e.key);
        expect(d.score, closeTo((w['score'] as num).toDouble(), 1e-9));
        expect(
          d.scoreDelta,
          closeTo((w['score_delta'] as num).toDouble(), 1e-9),
        );
        final tb = w['target_box'] as List?;
        if (tb != null) {
          expect(d.targetBox!.left, closeTo((tb[0] as num).toDouble(), 1e-9));
          expect(d.targetBox!.width, closeTo((tb[2] as num).toDouble(), 1e-9));
        }
      }
      final tilt = estimateTilt(p, params);
      if (c['tilt'] == null) {
        expect(tilt, isNull);
      } else {
        expect(tilt, closeTo((c['tilt'] as num).toDouble(), 1e-6));
      }
    }
  });

  test('no subject message and rotate hint', () {
    expect(
      decide(null, scorers['food_center']!, params, noun: '음식').message,
      '음식을 화면에 담아주세요',
    );
    expect(decide(null, scorers['wide_full']!, params).direction, 'NO_SUBJECT');
    expect(rotateHint(10, params), 'CW');
    expect(rotateHint(-10, params), 'CCW');
    expect(rotateHint(3, params), isNull);
    expect(rotateHint(null, params), isNull);
  });
}
