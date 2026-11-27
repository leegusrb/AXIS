import 'package:flutter/foundation.dart';

import '../api/models.dart';
import '../config.dart';
import 'stabilizer.dart';

/// T는 분석에 넘기는 프레임(서버: JPEG 바이트, 폰: FrameData).
class GuideController<T> extends ChangeNotifier {
  GuideController(this._analyze, {DateTime Function()? clock})
    : _clock = clock ?? DateTime.now;

  final Future<GuideResult> Function(T) _analyze;
  final DateTime Function() _clock;
  final GuideStabilizer _stabilizer = GuideStabilizer();

  bool _busy = false;
  bool _disposed = false;
  DateTime? _lastSent;
  int _failures = 0;
  int _epoch = 0; // reset() 이전에 보낸 요청의 응답을 버리기 위한 표식
  GuideResult? lastResult;

  GuideResult? get shown => _stabilizer.shown;
  String? get shownRotate => _stabilizer.shownRotate;
  bool get disconnected => _failures >= failureLimit;

  /// 인코딩 전에 확인해 비싼 변환을 건너뛸 수 있다.
  bool get readyForFrame =>
      !_disposed &&
      !_busy &&
      (_lastSent == null || _clock().difference(_lastSent!) >= frameInterval);

  Future<void> onFrame(T jpeg) async {
    if (!readyForFrame) return;
    _busy = true;
    _lastSent = _clock();
    final epoch = _epoch;
    try {
      final r = await _analyze(jpeg);
      if (_disposed || epoch != _epoch) return;
      lastResult = r;
      _failures = 0;
      _stabilizer.update(r, _clock());
    } catch (_) {
      if (!_disposed && epoch == _epoch) {
        _failures++;
        if (disconnected) _clear(); // 오래된 점수가 촬영 기록에 남지 않게 한다
      }
    } finally {
      _busy = false;
      if (!_disposed) notifyListeners();
    }
  }

  void _clear() {
    lastResult = null;
    _stabilizer.reset();
  }

  /// 마지막 결과와 안정화 상태를 지운다. 카메라 스트림을 다시 시작할 때 부른다.
  void reset() {
    _epoch++;
    _clear();
    if (!_disposed) notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    super.dispose();
  }
}
