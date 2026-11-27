import 'dart:ui';

import 'package:axis_app/api/models.dart';
import 'package:axis_app/widgets/guide_overlay.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

GuideResult result({Rect? subject, Rect? target}) => GuideResult(
  direction: 'LEFT',
  magnitude: 'SMALL',
  message: 'm',
  score: 0.5,
  scoreDelta: 0,
  confidence: 0.9,
  subjectBox: subject,
  targetBox: target,
  scorer: 'stat',
  modelVersion: 'v',
  timingMs: const {},
);

// 1x1 흰 PNG
const _png =
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGP4DwABAQEAG2fdOAAAAABJRU5ErkJggg==';

void main() {
  testWidgets('깊이 보기 창', (t) async {
    await t.pumpWidget(
      const MaterialApp(
        home: Scaffold(body: DepthPreview(base64Png: _png)),
      ),
    );
    expect(find.byType(Image), findsOneWidget);
    expect(find.text('깊이'), findsOneWidget);
    await t.pumpWidget(
      const MaterialApp(home: Scaffold(body: DepthPreview(base64Png: null))),
    );
    expect(find.byType(Image), findsNothing);
  });

  testWidgets('회전 안내 아이콘과 문구', (t) async {
    Widget hud(String? rotate) => MaterialApp(
      home: Scaffold(
        body: GuideHud(result: result(), rotate: rotate),
      ),
    );
    await t.pumpWidget(hud('CW'));
    expect(find.byIcon(Icons.rotate_right), findsOneWidget);
    expect(find.text('휴대폰을 시계 방향으로 살짝 돌리세요'), findsOneWidget);
    await t.pumpWidget(hud('CCW'));
    expect(find.byIcon(Icons.rotate_left), findsOneWidget);
    await t.pumpWidget(hud(null));
    expect(find.byIcon(Icons.rotate_right), findsNothing);
    expect(find.byIcon(Icons.rotate_left), findsNothing);
  });

  test('정규화 박스를 프리뷰 좌표로 변환', () {
    expect(
      toCanvasRect(
        const Rect.fromLTWH(0.25, 0.5, 0.5, 0.25),
        const Size(400, 800),
      ),
      const Rect.fromLTWH(100, 400, 200, 200),
    );
  });

  test('result가 null이면 아무것도 그리지 않는다', () {
    final recorder = PictureRecorder();
    final canvas = Canvas(recorder);
    const GuideOverlayPainter(null).paint(canvas, const Size(400, 800));
    expect(
      const GuideOverlayPainter(null)
          .shouldRepaint(const GuideOverlayPainter(null)),
      isFalse,
    );
    // 박스가 있는 결과는 예외 없이 그려진다.
    GuideOverlayPainter(
      result(
        subject: const Rect.fromLTWH(0.1, 0.1, 0.3, 0.5),
        target: const Rect.fromLTWH(0.3, 0.2, 0.3, 0.5),
      ),
    ).paint(canvas, const Size(400, 800));
    recorder.endRecording();
  });

  test('guideIcon 매핑', () {
    expect(guideIcon('LEFT'), Icons.arrow_back);
    expect(guideIcon('CLOSER'), Icons.zoom_in);
    expect(guideIcon('NO_SUBJECT'), Icons.person_search);
    expect(guideIcon('???'), isNull);
  });

  test('showTarget이 바뀌면 다시 그린다', () {
    final r = result(target: const Rect.fromLTWH(0.3, 0.2, 0.3, 0.5));
    expect(
      GuideOverlayPainter(r)
          .shouldRepaint(GuideOverlayPainter(r, showTarget: false)),
      isTrue,
    );
  });
}
