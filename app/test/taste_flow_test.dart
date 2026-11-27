import 'package:axis_app/api/axis_api.dart';
import 'package:axis_app/api/models.dart';
import 'package:axis_app/screens/camera_screen.dart';
import 'package:axis_app/screens/home_screen.dart';
import 'package:axis_app/screens/settings_screen.dart';
import 'package:axis_app/screens/taste_screen.dart';
import 'package:axis_app/state/app_state.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

class FakeApi implements AxisApi {
  FakeApi({this.failRecommend = false});
  final bool failRecommend;
  List<String>? liked;

  static const _samples = [
    StyleSample(url: '/m/a.jpg', author: '김작가', license: 'CC0'),
    StyleSample(url: '/m/b.jpg', author: '이작가', license: 'CC BY 4.0'),
    StyleSample(url: '/m/c.jpg', author: '박작가', license: 'PDM'),
  ];

  @override
  Future<Map<String, dynamic>> health() async => {'model_version': 'v-test'};
  @override
  Future<List<StyleInfo>> styles({String? category}) async => const [
    StyleInfo(
      id: 'wide_full',
      name: '여백 있는 전신',
      description: '',
      samples: _samples,
    ),
    StyleInfo(
      id: 'tight_close',
      name: '꽉 찬 클로즈업',
      description: '',
      samples: _samples,
    ),
    StyleInfo(
      id: 'thirds_upper',
      name: '삼분할 상반신',
      description: '',
      samples: _samples,
    ),
  ];
  @override
  Future<List<TastePhoto>> tastePhotos() async => [
    for (var i = 0; i < 12; i++) TastePhoto(id: 'p$i', url: '/m/p$i.jpg'),
  ];
  @override
  Future<List<StyleScore>> recommend(List<String> likedPhotoIds) async {
    liked = likedPhotoIds;
    if (failRecommend) throw AxisApiException(500, 'boom');
    return const [
      StyleScore(styleId: 'wide_full', score: 82.4),
      StyleScore(styleId: 'tight_close', score: 10),
      StyleScore(styleId: 'thirds_upper', score: 7.6),
    ];
  }

  @override
  String? token;
  @override
  dynamic noSuchMethod(Invocation invocation) => throw UnimplementedError();
  @override
  String mediaUrl(String url) => 'http://fake$url';
}

Future<AppState> makeState(FakeApi api) async {
  SharedPreferences.setMockInitialValues({});
  return AppState.load(apiFactory: (_) => api);
}

Widget app(AppState s, Widget home) => AppScope(
  notifier: s,
  child: MaterialApp(home: home),
);

const errorText = '서버에 연결할 수 없어요. 설정에서 주소를 확인하세요';

void main() {
  testWidgets('취향 고르기 → 추천', (tester) async {
    final api = FakeApi();
    final state = await makeState(api);
    await tester.pumpWidget(app(state, const TasteScreen()));
    await tester.pumpAndSettle();

    ButtonStyleButton button() =>
        tester.widget(find.widgetWithText(FilledButton, '추천 받기'));
    expect(button().onPressed, isNull);

    await tester.tap(find.byKey(const ValueKey('taste-p0')));
    await tester.tap(find.byKey(const ValueKey('taste-p3')));
    await tester.pump();
    expect(button().onPressed, isNotNull);
    expect(find.byIcon(Icons.check_circle), findsNWidgets(2));

    await tester.tap(find.widgetWithText(FilledButton, '추천 받기'));
    await tester.pumpAndSettle();

    expect(api.liked, ['p0', 'p3']);
    expect(find.text("당신에게 맞는 작가: '여백 있는 전신' (82%)"), findsOneWidget);
    expect(find.text('사진: 김작가 · CC0'), findsOneWidget);
    expect(find.text('이 작가로 촬영'), findsOneWidget);

    await tester.tap(find.widgetWithText(ChoiceChip, '꽉 찬 클로즈업'));
    await tester.pumpAndSettle();
    expect(state.selectedStyle?.id, 'tight_close');
    expect(find.textContaining("'꽉 찬 클로즈업' ("), findsOneWidget);

    await tester.tap(find.text('이 작가로 촬영'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
    expect(find.byType(CameraScreen), findsOneWidget);
  });

  testWidgets('추천 실패 시 SnackBar', (tester) async {
    final state = await makeState(FakeApi(failRecommend: true));
    await tester.pumpWidget(app(state, const TasteScreen()));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('taste-p1')));
    await tester.pump();
    await tester.tap(find.widgetWithText(FilledButton, '추천 받기'));
    await tester.pumpAndSettle();
    expect(find.text(errorText), findsOneWidget);
    expect(find.text('이 작가로 촬영'), findsNothing);
  });

  testWidgets('설정: 주소 저장과 연결 확인', (tester) async {
    final state = await makeState(FakeApi());
    await tester.pumpWidget(app(state, const SettingsScreen()));
    await tester.enterText(find.byType(TextField), 'http://10.0.0.5:8000');
    await tester.testTextInput.receiveAction(TextInputAction.done);
    await tester.pump();
    expect(state.serverUrl, 'http://10.0.0.5:8000');
    expect(
      (await SharedPreferences.getInstance()).getString('serverUrl'),
      'http://10.0.0.5:8000',
    );

    await tester.tap(find.text('연결 확인'));
    await tester.pumpAndSettle();
    expect(find.textContaining('v-test'), findsOneWidget);
  });

  testWidgets('홈 문구와 deviceId 유지', (tester) async {
    final state = await makeState(FakeApi());
    expect(state.deviceId, hasLength(36));
    final again = await AppState.load(apiFactory: (_) => FakeApi());
    expect(again.deviceId, state.deviceId);
    await tester.pumpWidget(app(state, const HomeScreen()));
    expect(find.text('촬영 시작'), findsOneWidget);
    expect(find.text('촬영 기록'), findsOneWidget);
    expect(find.byIcon(Icons.settings), findsOneWidget);
  });
}
