// ai/axis_ai/guide.py를 옮긴 것. direction은 사용자가 카메라를 움직일 방향.
import 'dart:math' as math;
import 'dart:ui' show Rect;

import 'geometry.dart';
import 'scorers.dart';

const _messages = {
  'LEFT': ('카메라를 왼쪽으로 조금 옮기세요', '카메라를 왼쪽으로 옮기세요'),
  'RIGHT': ('카메라를 오른쪽으로 조금 옮기세요', '카메라를 오른쪽으로 옮기세요'),
  'UP': ('카메라를 위로 조금 옮기세요', '카메라를 위로 옮기세요'),
  'DOWN': ('카메라를 아래로 조금 옮기세요', '카메라를 아래로 옮기세요'),
  'CLOSER': ('인물에게 조금 다가가세요', '인물에게 다가가세요'),
  'FARTHER': ('조금 뒤로 물러나세요', '뒤로 물러나세요'),
};
const okMessage = '좋아요! 지금 찍으세요';
const okTiltedMessage = '좋아요! 수평만 맞추면 완벽해요';
const rotateMessages = {
  'CW': '휴대폰을 시계 방향으로 살짝 돌리세요',
  'CCW': '휴대폰을 반시계 방향으로 살짝 돌리세요',
};
const _maxTargetSteps = 5;

String messageFor(String direction, String magnitude, {String noun = '인물'}) {
  if (direction == 'OK') return okMessage;
  if (direction == 'NO_SUBJECT') return '$noun을 화면에 담아주세요';
  final m = _messages[direction]!;
  final msg = magnitude == 'SMALL' ? m.$1 : m.$2;
  return noun == '인물' ? msg : msg.replaceAll('인물에게', '$noun에');
}

class LocalDecision {
  const LocalDecision(
    this.direction,
    this.magnitude,
    this.message,
    this.score,
    this.scoreDelta,
    this.confidence,
    this.subjectBox,
    this.targetBox,
  );
  final String direction, magnitude, message;
  final double score, scoreDelta, confidence;
  final Rect? subjectBox, targetBox;
}

Map<String, (double, double, double)> moves(LocalParams p) => {
  'LEFT': (p.moveStep, 0.0, 1.0),
  'RIGHT': (-p.moveStep, 0.0, 1.0),
  'UP': (0.0, p.moveStep, 1.0),
  'DOWN': (0.0, -p.moveStep, 1.0),
  'CLOSER': (0.0, 0.0, p.zoomIn),
  'FARTHER': (0.0, 0.0, p.zoomOut),
};

double _score(Subject s, LocalScorer sc, LocalParams p) =>
    sc.score(computeFeatures(s, p));

Rect _targetBox(
  Subject s,
  (double, double, double) m,
  LocalScorer sc,
  LocalParams p,
) {
  var cur = s;
  var best = _score(s, sc, p);
  for (var i = 0; i < _maxTargetSteps; i++) {
    final nxt = transformSubject(cur, m.$1, m.$2, m.$3, p);
    if (nxt == null) break;
    final v = _score(nxt, sc, p);
    if (v <= best) break;
    cur = nxt;
    best = v;
  }
  return cur.box;
}

LocalDecision decide(
  Subject? s,
  LocalScorer sc,
  LocalParams p, {
  String noun = '인물',
}) {
  if (s == null) {
    return LocalDecision(
      'NO_SUBJECT',
      'SMALL',
      messageFor('NO_SUBJECT', 'SMALL', noun: noun),
      0,
      0,
      0,
      null,
      null,
    );
  }
  final s0 = _score(s, sc, p);
  final mv = moves(p);
  final scores = <String, double>{};
  for (final e in mv.entries) {
    final c = transformSubject(s, e.value.$1, e.value.$2, e.value.$3, p);
    if (c != null) scores[e.key] = _score(c, sc, p);
  }
  if (scores.isEmpty) {
    if (s0 >= p.okScore) {
      return LocalDecision(
        'OK',
        'SMALL',
        okMessage,
        s0,
        0,
        s.conf,
        s.box,
        s.box,
      );
    }
    return LocalDecision(
      'FARTHER',
      'SMALL',
      messageFor('FARTHER', 'SMALL', noun: noun),
      s0,
      0,
      s.conf * 0.5,
      s.box,
      s.box,
    );
  }
  // 멀리 떨어져 점수가 거의 0이어도 방향이 보이도록 로그 점수 상승폭으로 고른다(파이썬과 같은 순서로 비교).
  final log0 = math.log(math.max(s0, 1e-12));
  String? best;
  var bestGain = double.negativeInfinity;
  for (final e in scores.entries) {
    final g = math.log(math.max(e.value, 1e-12)) - log0;
    if (g > bestGain) {
      best = e.key;
      bestGain = g;
    }
  }
  final delta = scores[best]! - s0;
  if (delta < p.okDelta && s0 >= p.okScore) {
    return LocalDecision(
      'OK',
      'SMALL',
      okMessage,
      s0,
      delta,
      s.conf,
      s.box,
      s.box,
    );
  }
  final magnitude = bestGain >= p.largeLogGain ? 'LARGE' : 'SMALL';
  final conf = bestGain <= 0 ? s.conf * 0.5 : s.conf;
  return LocalDecision(
    best!,
    magnitude,
    messageFor(best, magnitude, noun: noun),
    s0,
    delta,
    conf,
    s.box,
    _targetBox(s, mv[best]!, sc, p),
  );
}
