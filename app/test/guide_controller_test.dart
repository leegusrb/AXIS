import 'dart:async';
import 'dart:typed_data';

import 'package:axis_app/api/models.dart';
import 'package:axis_app/guide/guide_controller.dart';
import 'package:flutter_test/flutter_test.dart';

GuideResult res(String dir) => GuideResult(
  direction: dir,
  magnitude: 'SMALL',
  message: dir,
  score: 0.5,
  scoreDelta: 0,
  confidence: 0.9,
  subjectBox: null,
  targetBox: null,
  scorer: 'stat',
  modelVersion: 'v',
  timingMs: const {},
);

final jpeg = Uint8List.fromList([1]);

void main() {
  late DateTime now;
  setUp(() => now = DateTime(2026, 1, 1));

  test(
    'dispose while a request is pending does not throw afterwards',
    () async {
      final c = Completer<GuideResult>();
      final ctl = GuideController((_) => c.future, clock: () => now);
      final f = ctl.onFrame(jpeg);
      ctl.dispose();
      c.complete(res('LEFT'));
      await f;
      expect(ctl.lastResult, isNull);
    },
  );

  test('controller sends one request at a time', () async {
    var calls = 0;
    final c = Completer<GuideResult>();
    final ctl = GuideController((_) {
      calls++;
      return c.future;
    }, clock: () => now);
    final f = ctl.onFrame(jpeg);
    now = now.add(const Duration(seconds: 5));
    expect(ctl.readyForFrame, isFalse);
    await ctl.onFrame(jpeg);
    await ctl.onFrame(jpeg);
    expect(calls, 1);
    c.complete(res('LEFT'));
    await f;
    expect(ctl.lastResult!.direction, 'LEFT');
    expect(ctl.readyForFrame, isTrue);
  });

  test('controller throttles to frameInterval', () async {
    var calls = 0;
    final ctl = GuideController((_) async {
      calls++;
      return res('LEFT');
    }, clock: () => now);
    await ctl.onFrame(jpeg);
    now = now.add(const Duration(milliseconds: 399));
    expect(ctl.readyForFrame, isFalse);
    await ctl.onFrame(jpeg);
    expect(calls, 1);
    now = now.add(const Duration(milliseconds: 1));
    expect(ctl.readyForFrame, isTrue);
    await ctl.onFrame(jpeg);
    expect(calls, 2);
    expect(ctl.shown!.direction, 'LEFT'); // two consecutive LEFT
  });

  test('controller marks disconnected after 3 failures and recovers', () async {
    var fail = true;
    final ctl = GuideController((_) async {
      if (fail) throw TimeoutException('slow');
      return res('RIGHT');
    }, clock: () => now);
    var notified = 0;
    ctl.addListener(() => notified++);
    for (var i = 0; i < 3; i++) {
      expect(ctl.disconnected, isFalse);
      await ctl.onFrame(jpeg);
      now = now.add(const Duration(milliseconds: 400));
    }
    expect(ctl.disconnected, isTrue);
    expect(notified, 3);
    fail = false;
    await ctl.onFrame(jpeg);
    expect(ctl.disconnected, isFalse);
    expect(ctl.lastResult!.direction, 'RIGHT');
  });

  test('failed request does not block next frame', () async {
    final ctl = GuideController(
      (_) async => throw Exception('x'),
      clock: () => now,
    );
    await ctl.onFrame(jpeg);
    now = now.add(const Duration(milliseconds: 400));
    expect(ctl.readyForFrame, isTrue);
  });

  test(
    'reset clears lastResult and shown, and drops an in-flight response',
    () async {
      final ctl = GuideController((_) async => res('LEFT'), clock: () => now);
      for (var i = 0; i < 3; i++) {
        now = now.add(const Duration(seconds: 1));
        await ctl.onFrame(jpeg);
      }
      expect(ctl.lastResult, isNotNull);
      expect(ctl.shown, isNotNull);
      ctl.reset();
      expect(ctl.lastResult, isNull);
      expect(ctl.shown, isNull);

      final c = Completer<GuideResult>();
      final slow = GuideController((_) => c.future, clock: () => now);
      final f = slow.onFrame(jpeg);
      slow.reset();
      c.complete(res('LEFT'));
      await f;
      expect(slow.lastResult, isNull);
    },
  );

  test('going disconnected clears the stale result', () async {
    var fail = false;
    final ctl = GuideController((_) async {
      if (fail) throw Exception('down');
      return res('LEFT');
    }, clock: () => now);
    now = now.add(const Duration(seconds: 1));
    await ctl.onFrame(jpeg);
    expect(ctl.lastResult, isNotNull);
    fail = true;
    for (var i = 0; i < 3; i++) {
      now = now.add(const Duration(seconds: 1));
      await ctl.onFrame(jpeg);
    }
    expect(ctl.disconnected, isTrue);
    expect(ctl.lastResult, isNull);
    expect(ctl.shown, isNull);
  });
}
