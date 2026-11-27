import 'dart:convert';
import 'dart:typed_data';

import 'package:axis_app/api/axis_api.dart';
import 'package:axis_app/api/models.dart';
import 'package:axis_app/screens/history_screen.dart';
import 'package:axis_app/screens/result_screen.dart';
import 'package:axis_app/state/app_state.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

class FakeApi implements AxisApi {
  FakeApi(this._shots);
  final List<Shot> _shots;

  @override
  String? token;
  final asked = <String?>[];
  @override
  Future<List<Shot>> shots({String? styleId}) async {
    asked.add(styleId);
    return _shots
        .where((s) => styleId == null || s.styleId == styleId)
        .toList();
  }

  @override
  Future<List<StyleInfo>> styles({String? category}) async => const [
    StyleInfo(id: 'wide_full', name: '여백 있는 전신', description: '', samples: []),
  ];
  @override
  String mediaUrl(String url) => 'http://fake$url';
  @override
  dynamic noSuchMethod(Invocation invocation) => throw UnimplementedError();
}

Future<AppState> makeState(AxisApi api) async {
  SharedPreferences.setMockInitialValues({'token': 't', 'username': 'u'});
  return AppState.load(apiFactory: (_) => api);
}

Widget app(AppState s, Widget home) => AppScope(
  notifier: s,
  child: MaterialApp(home: home),
);

void main() {
  testWidgets('기록: 스타일 이름, 점수, 날짜', (tester) async {
    final created = DateTime(2026, 11, 16, 9, 5);
    final state = await makeState(
      FakeApi([
        Shot(
          id: '1',
          styleId: 'wide_full',
          score: 0.824,
          guideUsed: true,
          url: '/shots/1.jpg',
          createdAt: created.toUtc().toIso8601String(),
        ),
        Shot(
          id: '2',
          styleId: 'unknown_style',
          score: 0.5,
          guideUsed: false,
          url: '/shots/2.jpg',
          createdAt: created.toUtc().toIso8601String(),
        ),
      ]),
    );
    await tester.pumpWidget(app(state, const HistoryScreen()));
    await tester.pumpAndSettle();
    expect(find.widgetWithText(ListTile, '여백 있는 전신'), findsOneWidget);
    expect(find.widgetWithText(ListTile, 'unknown_style'), findsOneWidget);
    expect(find.text('82점 · 2026.11.16 09:05'), findsOneWidget);
    expect(find.text('50점 · 2026.11.16 09:05'), findsOneWidget);
  });

  testWidgets('작가 필터 칩', (tester) async {
    Shot shot(String id, String style) => Shot(
      id: id,
      styleId: style,
      score: 0.5,
      guideUsed: true,
      url: '/s/$id.jpg',
      createdAt: '2026-11-28T00:00:00Z',
    );
    final api = FakeApi([
      shot('1', 'wide_full'),
      shot('2', 'sym_back'),
      shot('3', 'wide_full'),
    ]);
    final state = await makeState(api);
    await tester.pumpWidget(app(state, const HistoryScreen()));
    await tester.pumpAndSettle();
    expect(find.byType(ChoiceChip), findsNWidgets(3)); // 전체 + 작가 2명
    expect(find.byType(ListTile), findsNWidgets(3));
    await tester.tap(find.widgetWithText(ChoiceChip, 'sym_back'));
    await tester.pumpAndSettle();
    expect(api.asked.last, 'sym_back');
    expect(find.byType(ListTile), findsOneWidget);
    expect(find.byType(ChoiceChip), findsNWidgets(3)); // 칩은 그대로
    await tester.tap(find.widgetWithText(ChoiceChip, '전체'));
    await tester.pumpAndSettle();
    expect(api.asked.last, isNull);
    expect(find.byType(ListTile), findsNWidgets(3));
  });

  testWidgets('기록이 없으면 안내 문구', (tester) async {
    final state = await makeState(FakeApi([]));
    await tester.pumpWidget(app(state, const HistoryScreen()));
    await tester.pumpAndSettle();
    expect(find.text('아직 찍은 사진이 없어요'), findsOneWidget);
  });

  testWidgets('결과 화면: 스타일 이름과 정수 점수', (tester) async {
    final state = await makeState(FakeApi([]));
    state.selectStyle(
      const StyleInfo(
        id: 'wide_full',
        name: '여백 있는 전신',
        description: '',
        samples: [],
      ),
    );
    // 1x1 투명 PNG
    final png = Uint8List.fromList(
      base64Decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
      ),
    );
    await tester.pumpWidget(
      app(
        state,
        ResultScreen(
          shot: const Shot(
            id: '1',
            styleId: 'wide_full',
            score: 0.826,
            guideUsed: true,
            url: '/shots/1.jpg',
            createdAt: '2026-11-16T00:00:00Z',
          ),
          photo: png,
        ),
      ),
    );
    expect(find.text('여백 있는 전신'), findsOneWidget);
    expect(find.text('구도 점수 83점'), findsOneWidget);
  });
}
