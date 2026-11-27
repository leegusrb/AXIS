import 'package:axis_app/api/axis_api.dart';
import 'package:axis_app/screens/settings_screen.dart';
import 'package:axis_app/state/app_state.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

class FakeApi implements AxisApi {
  @override
  String? token;
  @override
  Future<Map<String, dynamic>> health() async => {
    'model_version': 'v4',
    'scorers': ['head', 'stat'],
  };
  @override
  dynamic noSuchMethod(Invocation invocation) => throw UnimplementedError();
}

void main() {
  testWidgets('scorer choices come from server and persist', (t) async {
    SharedPreferences.setMockInitialValues({});
    final state = await AppState.load(apiFactory: (_) => FakeApi());
    await t.pumpWidget(
      AppScope(
        notifier: state,
        child: const MaterialApp(home: SettingsScreen()),
      ),
    );
    await t.pumpAndSettle();
    expect(find.text('통계'), findsOneWidget);
    expect(find.text('통합 가이드 모델(Guide Head)'), findsOneWidget);
    expect(find.text('순위 학습'), findsNothing);
    await t.tap(find.text('통합 가이드 모델(Guide Head)'));
    await t.pumpAndSettle();
    expect(state.scorer, 'head');
    expect((await SharedPreferences.getInstance()).getString('scorer'), 'head');
  });

  test('scorerLabel', () {
    expect(scorerLabel('stat'), '통계');
    expect(scorerLabel('rank'), '순위 학습');
    expect(scorerLabel('head'), '통합 가이드 모델(Guide Head)');
    expect(scorerLabel('rule'), '규칙');
  });
}
