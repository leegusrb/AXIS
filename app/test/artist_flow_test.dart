import 'package:axis_app/api/axis_api.dart';
import 'package:axis_app/api/models.dart';
import 'package:axis_app/main.dart';
import 'package:axis_app/screens/camera_screen.dart';
import 'package:axis_app/screens/taste_screen.dart';
import 'package:axis_app/state/app_state.dart';
import 'package:axis_app/widgets/composition_diagram.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _all = [
  StyleInfo(
    id: 'wide_full',
    name: '작가 A — 여백과 인물 중심',
    description: '전신과 여백',
    samples: [StyleSample(url: '/m/a.jpg', author: '김작가', license: 'CC0')],
    diagramBox: Rect.fromLTWH(0.4, 0.15, 0.2, 0.7),
  ),
  StyleInfo(
    id: 'sym_back',
    name: '작가 B — 대칭과 배경 강조',
    description: '가운데 작게',
    samples: [],
    kind: 'rule',
    diagramBox: Rect.fromLTWH(0.45, 0.33, 0.1, 0.5),
  ),
  StyleInfo(
    id: 'food_center',
    name: '작가 E — 가운데 크게',
    description: '음식 가운데',
    samples: [],
    category: 'food',
    kind: 'rule',
    diagramBox: Rect.fromLTWH(0.2, 0.2, 0.6, 0.6),
  ),
];

class FakeApi implements AxisApi {
  @override
  String? token;
  final asked = <String?>[];

  @override
  Future<List<Category>> categories() async => const [
    Category(id: 'person', name: '인물'),
    Category(id: 'food', name: '음식'),
  ];
  @override
  Future<List<StyleInfo>> styles({String? category}) async {
    asked.add(category);
    return _all
        .where((s) => category == null || s.category == category)
        .toList();
  }

  @override
  Future<List<TastePhoto>> tastePhotos() async => const [];
  @override
  String mediaUrl(String url) => 'http://fake$url';
  @override
  dynamic noSuchMethod(Invocation invocation) => throw UnimplementedError();
}

Future<AppState> pumpApp(WidgetTester t, FakeApi api) async {
  SharedPreferences.setMockInitialValues({'token': 't', 'username': 'u'});
  final state = await AppState.load(apiFactory: (_) => api);
  await t.pumpWidget(AxisApp(state: state));
  await t.pumpAndSettle();
  return state;
}

void main() {
  testWidgets('food category shows only food artists', (t) async {
    final api = FakeApi();
    final state = await pumpApp(t, api);
    await t.tap(find.text('촬영 시작'));
    await t.pumpAndSettle();
    expect(find.text('인물'), findsOneWidget);
    expect(find.text('음식'), findsOneWidget);
    await t.tap(find.text('음식'));
    await t.pumpAndSettle();
    expect(state.selectedCategory, 'food');
    expect(api.asked.last, 'food');
    expect(find.text('작가 E — 가운데 크게'), findsOneWidget);
    expect(find.text('작가 A — 여백과 인물 중심'), findsNothing);
    expect(find.text('나에게 맞는 작가 찾기'), findsNothing);
    expect(find.text('규칙 기반'), findsOneWidget);
    expect(find.byType(CompositionDiagram), findsOneWidget);
  });

  testWidgets('person artist card opens camera with that artist', (t) async {
    final state = await pumpApp(t, FakeApi());
    await t.tap(find.text('촬영 시작'));
    await t.pumpAndSettle();
    await t.tap(find.text('인물'));
    await t.pumpAndSettle();
    expect(find.text('나에게 맞는 작가 찾기'), findsOneWidget);
    expect(find.byType(CompositionDiagram), findsNWidgets(2));
    await t.tap(find.text('작가 B — 대칭과 배경 강조'));
    await t.pump();
    await t.pump(const Duration(milliseconds: 500));
    expect(state.selectedStyle?.id, 'sym_back');
    expect(find.byType(CameraScreen), findsOneWidget);
  });

  testWidgets('find my artist opens taste screen', (t) async {
    await pumpApp(t, FakeApi());
    await t.tap(find.text('촬영 시작'));
    await t.pumpAndSettle();
    await t.tap(find.text('인물'));
    await t.pumpAndSettle();
    await t.tap(find.text('나에게 맞는 작가 찾기'));
    await t.pumpAndSettle();
    expect(find.byType(TasteScreen), findsOneWidget);
  });

  testWidgets('composition diagram paints', (t) async {
    await t.pumpWidget(
      const Directionality(
        textDirection: TextDirection.ltr,
        child: Center(
          child: SizedBox(
            width: 90,
            child: CompositionDiagram(box: Rect.fromLTWH(0.2, 0.2, 0.6, 0.6)),
          ),
        ),
      ),
    );
    expect(t.getSize(find.byType(CompositionDiagram)), const Size(90, 120));
  });
}
