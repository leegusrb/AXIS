import '../api/models.dart';
import '../config.dart' as cfg;

/// 서버 결과를 화면에 보일 안내로 안정화한다(순수 Dart).
class GuideStabilizer {
  GuideStabilizer({
    this.stableRepeats = cfg.stableRepeats,
    this.minHold = cfg.minHold,
    this.minConfidence = cfg.minConfidence,
    this.noSubjectRepeats = cfg.noSubjectRepeats,
  });

  final int stableRepeats;
  final Duration minHold;
  final double minConfidence;
  final int noSubjectRepeats;

  GuideResult? _shown;
  DateTime? _shownSince;
  String? _candidate;
  int _candidateCount = 0;
  int _noSubjectCount = 0;
  String? _rotate;
  String? _rotateCandidate;
  int _rotateCount = 0;

  GuideResult? get shown => _shown;

  /// 화면에 보일 회전 안내(CW/CCW). 같은 값이 stableRepeats번 연달아 와야 바뀐다.
  String? get shownRotate => _rotate;

  void _updateRotate(String? r) {
    if (r == _rotate) {
      _rotateCandidate = null;
      _rotateCount = 0;
      return;
    }
    if (r == _rotateCandidate) {
      _rotateCount++;
    } else {
      _rotateCandidate = r;
      _rotateCount = 1;
    }
    if (_rotateCount >= stableRepeats) {
      _rotate = r;
      _rotateCandidate = null;
      _rotateCount = 0;
    }
  }

  GuideResult? update(GuideResult r, DateTime now) {
    final noSubject = r.direction == 'NO_SUBJECT';
    if (!noSubject && r.confidence < minConfidence) return _shown;
    _updateRotate(noSubject ? null : r.rotate);

    if (noSubject) {
      _noSubjectCount++;
      if (_noSubjectCount >= noSubjectRepeats) _show(r, now);
      return _shown;
    }
    _noSubjectCount = 0;

    if (_shown != null && r.direction == _shown!.direction) {
      _shown = r; // 최신 점수·박스로 갱신, 유지 시간은 그대로
      _candidate = null;
      _candidateCount = 0;
      return _shown;
    }

    if (_candidate == r.direction) {
      _candidateCount++;
    } else {
      _candidate = r.direction;
      _candidateCount = 1;
    }
    if (_candidateCount >= stableRepeats &&
        (_shown == null || now.difference(_shownSince!) >= minHold)) {
      _show(r, now);
    }
    return _shown;
  }

  /// 화면에 보이던 안내와 누적 상태를 모두 지운다(스트림 재시작, 연결 끊김).
  void reset() {
    _shown = null;
    _shownSince = null;
    _candidate = null;
    _candidateCount = 0;
    _noSubjectCount = 0;
    _rotate = null;
    _rotateCandidate = null;
    _rotateCount = 0;
  }

  void _show(GuideResult r, DateTime now) {
    if (_shown?.direction != r.direction) _shownSince = now;
    _shown = r;
    _candidate = null;
    _candidateCount = 0;
  }
}
