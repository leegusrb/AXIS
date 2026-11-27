import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:axis_app/local/yolo_decode.dart';
import 'package:flutter_test/flutter_test.dart';

/// (채널, n) 행 우선 출력을 만든다. boxes는 320 입력 기준 0–1 (cx, cy, w, h).
Float32List fakeOut(int ch, int n, List<(List<double>, List<double>)> rows) {
  final out = Float32List(ch * n);
  for (var j = 0; j < rows.length; j++) {
    final vals = [...rows[j].$1, ...rows[j].$2];
    for (var c = 0; c < vals.length; c++) {
      out[c * n + j] = vals[c];
    }
  }
  return out;
}

const lbSquare = Letterbox(
  scale: 1,
  padX: 0,
  padY: 0,
  srcW: 320,
  srcH: 320,
  size: 320,
);

void main() {
  test('decode matches python', () {
    final fx =
        (jsonDecode(File('test/fixtures/local_parity.json').readAsStringSync())
                as Map<String, dynamic>)['pose_frame']
            as Map<String, dynamic>;
    final lbj = fx['letterbox'] as Map<String, dynamic>;
    final lb = Letterbox(
      scale: (lbj['scale'] as num).toDouble(),
      padX: (lbj['pad_x'] as num).toDouble(),
      padY: (lbj['pad_y'] as num).toDouble(),
      srcW: lbj['src_w'] as int,
      srcH: lbj['src_h'] as int,
      size: 320,
    );
    final out = File('test/fixtures/pose_out.bin')
        .readAsBytesSync()
        .buffer
        .asFloat32List();
    final people = decodePose(out, fx['n'] as int, lb);
    people.sort((a, b) => b.conf.compareTo(a.conf));
    final want = (fx['top_box'] as List)
        .map((e) => (e as num).toDouble())
        .toList();
    final top = people.first;
    expect(top.conf, closeTo((fx['top_conf'] as num).toDouble(), 1e-6));
    expect(top.box.left, closeTo(want[0], 1e-4));
    expect(top.box.top, closeTo(want[1], 1e-4));
    expect(top.box.width, closeTo(want[2], 1e-4));
    expect(top.keypoints.length, 17);
    expect(top.keypoints[0][2], greaterThan(0.5)); // 코
  });

  test('nms keeps one', () {
    final kp = List.filled(51, 0.5);
    final out = fakeOut(56, 4, [
      ([0.5, 0.5, 0.4, 0.6, 0.9], kp),
      ([0.51, 0.5, 0.4, 0.6, 0.8], kp),
      ([0.2, 0.2, 0.1, 0.1, 0.1], kp), // 신뢰도 미달
    ]);
    final people = decodePose(out, 4, lbSquare);
    expect(people.length, 1);
    expect(people.single.conf, closeTo(0.9, 1e-6));
  });

  test('decode clamps to frame', () {
    final out = fakeOut(56, 1, [
      ([0.05, 0.5, 0.3, 0.4, 0.9], List.filled(51, 0.5)),
    ]);
    final b = decodePose(out, 1, lbSquare).single.box;
    expect(b.left, 0);
    expect(b.width, closeTo(0.2, 1e-6));
  });

  test('letterbox is undone', () {
    // 640×320 원본을 320에 넣으면 scale 0.5, 위아래 80 padding
    const lb = Letterbox(
      scale: 0.5,
      padX: 0,
      padY: 80,
      srcW: 640,
      srcH: 320,
      size: 320,
    );
    final out = fakeOut(56, 1, [
      ([0.5, 0.5, 0.25, 0.25, 0.9], List.filled(51, 0.5)),
    ]);
    final b = decodePose(out, 1, lb).single.box;
    expect(b.left, closeTo(0.375, 1e-6));
    expect(b.top, closeTo(0.25, 1e-6));
    expect(b.height, closeTo(0.5, 1e-6));
  });

  test('food box whose top class is not food is dropped; nms per class', () {
    final classes = List<String?>.filled(80, null)
      ..[41] = 'cup'
      ..[45] = 'bowl';
    List<double> scores(Map<int, double> m) {
      final l = List.filled(80, 0.0);
      m.forEach((k, v) => l[k] = v);
      return l;
    }

    final out = fakeOut(84, 4, [
      ([0.5, 0.5, 0.9, 0.6], scores({60: 0.8, 45: 0.3})), // 식탁이 1등 -> 버림
      ([0.3, 0.3, 0.2, 0.2], scores({41: 0.7})), // 컵
      ([0.31, 0.3, 0.2, 0.2], scores({45: 0.6})), // 컵과 겹친 그릇 -> 다른 클래스라 남음
    ]);
    final food = decodeFood(out, 4, lbSquare, classes);
    expect(food.map((f) => f.label).toList()..sort(), ['bowl', 'cup']);
  });

  test('food keeps food classes only', () {
    final classes = List<String?>.filled(80, null)..[41] = 'cup';
    List<double> scores(int k, double v) => List.filled(80, 0.0)..[k] = v;
    final out = fakeOut(84, 3, [
      ([0.3, 0.3, 0.2, 0.2], scores(41, 0.8)),
      ([0.7, 0.7, 0.2, 0.2], scores(0, 0.95)), // 사람
    ]);
    final food = decodeFood(out, 3, lbSquare, classes);
    expect(food.length, 1);
    expect(food.single.label, 'cup');
    expect(food.single.keypoints.every((k) => k[2] == 0), isTrue);
  });
}
