// 폰에서 분석: TFLite YOLO + Dart 구도 계산. 서버의 RealPipeline(stat·규칙 경로)과 같은 결과를 낸다.
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show compute;
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../api/models.dart';
import '../camera/frame_encoder.dart';
import 'geometry.dart';
import 'guide.dart';
import 'preprocess.dart';
import 'scorers.dart';
import 'yolo_decode.dart';

const _assets = 'assets/local';

/// local.json: 상수, 학습값, 규칙, 작가 목록.
class LocalData {
  LocalData._(
    this.version,
    this.params,
    this.scorers,
    this.artists,
    this.categories,
    this.foodClasses,
  );

  factory LocalData.fromJsonString(String s) {
    final j = jsonDecode(s) as Map<String, dynamic>;
    final artists = [
      for (final a in j['artists'] as List)
        StyleInfo.fromJson({
          ...a as Map<String, dynamic>,
          'samples': const [],
          'model_version': j['version'],
        }),
    ];
    return LocalData._(
      j['version'] as String,
      LocalParams.fromJson(j['params'] as Map<String, dynamic>),
      loadScorers(j),
      artists,
      [
        for (final c in j['categories'] as List)
          Category.fromJson(c as Map<String, dynamic>),
      ],
      [for (final c in j['food_classes'] as List) c as String?],
    );
  }

  static Future<LocalData>? _cached;

  /// 앱에 넣은 local.json(모델 없이 작가 목록만 필요할 때도 쓴다).
  static Future<LocalData> load() => _cached ??= rootBundle
      .loadString('$_assets/local.json')
      .then(LocalData.fromJsonString);

  final String version;
  final LocalParams params;
  final Map<String, LocalScorer> scorers;
  final List<StyleInfo> artists;
  final List<Category> categories;
  final List<String?> foodClasses;
}

/// 검출된 피사체 목록으로 GuideResult를 만든다(모델과 무관한 순수 계산).
GuideResult localResult(
  List<Subject> found,
  StyleInfo style,
  LocalData data,
  Map<String, int> timing,
) {
  final sw = Stopwatch()..start();
  final food = style.category == 'food';
  final p = data.params;
  final subject = selectSubject(found, p);
  var d = decide(subject, data.scorers[style.id]!, p, noun: food ? '음식' : '인물');
  final tilt = subject == null ? null : estimateTilt(subject, p);
  final rotate = rotateHint(tilt, p);
  final message = rotate != null && d.direction == 'OK'
      ? okTiltedMessage
      : d.message;
  final guideMs = sw.elapsedMilliseconds;
  final t = {...timing, 'guide': guideMs};
  t['total'] = t.values.fold(0, (a, b) => a + b);
  final b = subject?.box;
  return GuideResult(
    direction: d.direction,
    magnitude: d.magnitude,
    message: message,
    score: d.score,
    scoreDelta: d.scoreDelta,
    confidence: d.confidence,
    subjectBox: d.subjectBox,
    targetBox: d.targetBox,
    scorer: style.kind == 'rule' ? 'rule' : 'stat',
    modelVersion: '${data.version}-local',
    timingMs: t,
    category: style.category,
    subject: b == null
        ? null
        : SubjectInfo(
            className: subject!.label,
            cx: b.center.dx,
            cy: b.center.dy,
            w: b.width,
            h: b.height,
            area: b.width * b.height,
            conf: subject.conf,
          ),
    rotationDeg: tilt,
    rotate: rotate,
  );
}

class LocalAnalyzer {
  LocalAnalyzer._(this.data, this._pose, this._food, this._poseN, this._foodN);

  final LocalData data;
  final IsolateInterpreter _pose, _food;
  final int _poseN, _foodN;

  static Future<LocalAnalyzer>? _instance;

  /// 모델 두 개와 local.json을 한 번만 읽는다.
  static Future<LocalAnalyzer> load() => _instance ??= _load();

  static Future<LocalAnalyzer> _load() async {
    final data = await LocalData.load();
    Future<(IsolateInterpreter, int)> open(String name) async {
      final it = await Interpreter.fromAsset(
        '$_assets/$name',
        options: InterpreterOptions()..threads = 4,
      );
      final n = it.getOutputTensor(0).shape.last;
      return (await IsolateInterpreter.create(address: it.address), n);
    }

    final (pose, pn) = await open('yolo11n-pose.tflite');
    final (food, fn) = await open('yolo11n.tflite');
    return LocalAnalyzer._(data, pose, food, pn, fn);
  }

  Future<GuideResult> analyze(FrameData frame, StyleInfo style) async {
    final sw = Stopwatch()..start();
    final input = await compute(prepareLocal, frame);
    final decodeMs = sw.elapsedMilliseconds;
    final food = style.category == 'food';
    final n = food ? _foodN : _poseN;
    final out = Float32List((food ? 84 : 56) * n);
    await (food ? _food : _pose).run(
      input.input.buffer.asUint8List(),
      out.buffer,
    );
    final found = food
        ? decodeFood(out, n, input.lb, data.foodClasses)
        : decodePose(out, n, input.lb);
    return localResult(found, style, data, {
      'decode': decodeMs,
      'pose': sw.elapsedMilliseconds - decodeMs,
    });
  }
}
