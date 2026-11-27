import 'package:axis_app/api/models.dart';
import 'package:axis_app/guide/stabilizer.dart';
import 'package:flutter_test/flutter_test.dart';

GuideResult res(
  String dir, {
  double conf = 0.8,
  double score = 0.5,
  String? rotate,
}) => GuideResult(
  rotate: rotate,
  direction: dir,
  magnitude: 'SMALL',
  message: dir,
  score: score,
  scoreDelta: 0,
  confidence: conf,
  subjectBox: null,
  targetBox: null,
  scorer: 'stat',
  modelVersion: 'v',
  timingMs: const {},
);

final t0 = DateTime(2026, 1, 1);
DateTime at(int ms) => t0.add(Duration(milliseconds: ms));

void main() {
  test('rotate needs two same values to change, both ways', () {
    final s = GuideStabilizer();
    s.update(res('LEFT', rotate: 'CW'), at(0));
    expect(s.shownRotate, isNull);
    s.update(res('LEFT', rotate: 'CW'), at(400));
    expect(s.shownRotate, 'CW');
    s.update(res('LEFT'), at(800));
    expect(s.shownRotate, 'CW');
    s.update(res('LEFT'), at(1200));
    expect(s.shownRotate, isNull);
    s.update(res('LEFT', rotate: 'CW'), at(1600));
    s.reset();
    expect(s.shownRotate, isNull);
  });

  test('rotate ignores low confidence frames', () {
    final s = GuideStabilizer();
    s.update(res('LEFT', rotate: 'CW', conf: 0.1), at(0));
    s.update(res('LEFT', rotate: 'CW', conf: 0.1), at(400));
    expect(s.shownRotate, isNull);
  });

  test('stabilizer needs two same directions', () {
    final s = GuideStabilizer();
    expect(s.update(res('LEFT'), at(0)), isNull);
    expect(s.update(res('LEFT'), at(400))?.direction, 'LEFT');
  });

  test('stabilizer does not switch on non-consecutive candidates', () {
    final s = GuideStabilizer();
    s.update(res('LEFT'), at(0));
    s.update(res('LEFT'), at(400));
    s.update(res('RIGHT'), at(2000));
    s.update(res('UP'), at(2400));
    expect(s.shown!.direction, 'LEFT');
  });

  test('stabilizer holds at least 800ms', () {
    final s = GuideStabilizer();
    s.update(res('LEFT'), at(0));
    s.update(res('LEFT'), at(400)); // shownSince = 400
    s.update(res('RIGHT'), at(500));
    expect(s.update(res('RIGHT'), at(900))!.direction, 'LEFT'); // 500ms held
    expect(s.update(res('RIGHT'), at(1200))!.direction, 'RIGHT'); // 800ms held
  });

  test('same direction refreshes score without resetting hold', () {
    final s = GuideStabilizer();
    s.update(res('LEFT', score: 0.1), at(0));
    s.update(res('LEFT', score: 0.2), at(400));
    expect(s.update(res('LEFT', score: 0.9), at(1000))!.score, 0.9);
    s.update(res('UP'), at(1100));
    // hold measured from 400, not from the refresh at 1000
    expect(s.update(res('UP'), at(1200))!.direction, 'UP');
  });

  test('stabilizer ignores low confidence', () {
    final s = GuideStabilizer();
    s.update(res('LEFT'), at(0));
    s.update(res('LEFT'), at(400));
    final low = res('RIGHT', conf: 0.3);
    expect(s.update(low, at(2000))!.direction, 'LEFT');
    expect(s.update(low, at(2400))!.direction, 'LEFT');
    // low-confidence results did not advance the candidate streak
    expect(s.update(res('RIGHT'), at(2800))!.direction, 'LEFT');
    expect(s.update(res('RIGHT'), at(3200))!.direction, 'RIGHT');
  });

  test('low confidence NO_SUBJECT still counts', () {
    final s = GuideStabilizer();
    for (var i = 0; i < 3; i++) {
      s.update(res('NO_SUBJECT', conf: 0.0), at(i * 400));
    }
    expect(s.shown!.direction, 'NO_SUBJECT');
  });

  test('stabilizer shows NO_SUBJECT after 3 in a row', () {
    final s = GuideStabilizer();
    s.update(res('LEFT'), at(0));
    s.update(res('LEFT'), at(400));
    expect(s.update(res('NO_SUBJECT'), at(500))!.direction, 'LEFT');
    expect(s.update(res('NO_SUBJECT'), at(600))!.direction, 'LEFT');
    expect(s.update(res('NO_SUBJECT'), at(700))!.direction, 'NO_SUBJECT');
  });

  test('another result resets the NO_SUBJECT streak', () {
    final s = GuideStabilizer();
    s.update(res('LEFT'), at(0));
    s.update(res('LEFT'), at(400));
    s.update(res('NO_SUBJECT'), at(1500));
    s.update(res('NO_SUBJECT'), at(1600));
    s.update(res('LEFT'), at(1700));
    s.update(res('NO_SUBJECT'), at(1800));
    s.update(res('NO_SUBJECT'), at(1900));
    expect(s.shown!.direction, 'LEFT');
  });
}
