import 'package:axis_app/api/axis_api.dart';
import 'package:axis_app/api/models.dart';
import 'package:axis_app/camera/frame_encoder.dart';
import 'package:axis_app/guide/guide_controller.dart';
import 'package:axis_app/main.dart';
import 'package:axis_app/screens/history_screen.dart';
import 'package:axis_app/screens/settings_screen.dart';
import 'package:axis_app/state/app_state.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// 서버가 꺼진 상태: 모든 호출이 실패한다.
class DeadApi implements AxisApi {
  @override
  String? token;
  @override
  Future<Map<String, dynamic>> health() async => throw Exception('no server');
  @override
  Future<List<Category>> categories() async => throw Exception('no server');
  @override
  Future<List<StyleInfo>> styles({String? category}) async =>
      throw Exception('no server');
  @override
  dynamic noSuchMethod(Invocation invocation) =>
      throw UnimplementedError('${invocation.memberName}');
  @override
  String mediaUrl(String url) => url;
}

Future<AppState> pump(
  WidgetTester t,
  Widget home, {
  Map<String, Object> prefs = const {},
}) async {
  SharedPreferences.setMockInitialValues(prefs);
  final state = await AppState.load(apiFactory: (_) => DeadApi());
  await t.pumpWidget(
    home is AxisApp
        ? AxisApp(state: state)
        : AppScope(
            notifier: state,
            child: MaterialApp(home: home),
          ),
  );
  await t.pumpAndSettle();
  return state;
}

void main() {
  testWidgets('guest start', (t) async {
    SharedPreferences.setMockInitialValues({});
    final state = await AppState.load(apiFactory: (_) => DeadApi());
    await t.pumpWidget(AxisApp(state: state));
    await t.pumpAndSettle();
    await t.tap(find.byKey(const Key('guestButton')));
    await t.pumpAndSettle();
    expect(find.text('촬영 시작'), findsOneWidget);
    expect(state.guest, isTrue);
    expect(state.analysis, 'local');
    expect((await SharedPreferences.getInstance()).getBool('guest'), isTrue);
  });

  testWidgets('local categories and artists without server', (t) async {
    SharedPreferences.setMockInitialValues({
      'guest': true,
      'analysis': 'local',
    });
    final state = await AppState.load(apiFactory: (_) => DeadApi());
    await t.pumpWidget(AxisApp(state: state));
    await t.pumpAndSettle();
    await t.tap(find.text('촬영 시작'));
    await t.pumpAndSettle();
    await t.tap(find.text('인물'));
    await t.pumpAndSettle();
    expect(find.text('작가 B — 대칭과 배경 강조'), findsOneWidget);
    expect(find.text('나에게 맞는 작가 찾기'), findsNothing); // 서버·로그인 필요
  });

  testWidgets('guest history message', (t) async {
    await pump(t, const HistoryScreen(), prefs: {'guest': true});
    expect(find.text('서버에 연결하면 기록을 볼 수 있어요'), findsOneWidget);
  });

  testWidgets('settings switches analysis place', (t) async {
    final state = await pump(t, const SettingsScreen(), prefs: {'guest': true});
    expect(state.analysis, 'server');
    await t.tap(find.text('폰에서 분석'));
    await t.pumpAndSettle();
    expect(state.analysis, 'local');
  });

  testWidgets('guest can go to login from settings', (t) async {
    final state = await pump(t, const SettingsScreen(), prefs: {'guest': true});
    await t.tap(find.text('로그인하기'));
    await t.pumpAndSettle();
    expect(state.guest, isFalse);
    expect(state.signedIn, isFalse);
  });

  test('local mode ignores server', () async {
    const r = GuideResult(
      direction: 'LEFT',
      magnitude: 'SMALL',
      message: 'm',
      score: 0.5,
      scoreDelta: 0,
      confidence: 0.9,
      subjectBox: null,
      targetBox: null,
      scorer: 'stat',
      modelVersion: 'v4-local',
      timingMs: {},
    );
    var t = DateTime(2026);
    final g = GuideController<FrameData>((_) async => r, clock: () => t);
    const f = FrameData(
      width: 1,
      height: 1,
      format: FrameFormat.bgra8888,
      planes: [],
      rotationDegrees: 0,
    );
    await g.onFrame(f);
    t = t.add(const Duration(seconds: 1));
    await g.onFrame(f);
    expect(g.lastResult, same(r));
    expect(g.shown?.direction, 'LEFT');
    expect(g.disconnected, isFalse);
  });
}
