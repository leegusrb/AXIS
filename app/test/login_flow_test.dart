import 'package:axis_app/api/axis_api.dart';
import 'package:axis_app/api/models.dart';
import 'package:axis_app/main.dart';
import 'package:axis_app/state/app_state.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

class FakeApi implements AxisApi {
  FakeApi({this.shotsStatus});
  final int? shotsStatus;
  @override
  String? token;
  String? signedUp;

  @override
  Future<AuthResult> login(String username, String password) async {
    if (password != 'secret1') {
      throw AxisApiException(401, '{"detail":"아이디나 비밀번호가 틀려요"}');
    }
    return AuthResult(token: 'tok-$username', username: username);
  }

  @override
  Future<AuthResult> signup(String username, String password) async {
    signedUp = username;
    return AuthResult(token: 'tok-$username', username: username);
  }

  @override
  Future<List<Shot>> shots({String? styleId}) async {
    if (shotsStatus != null) {
      throw AxisApiException(shotsStatus!, '{"detail":"로그인이 필요해요"}');
    }
    return const [];
  }

  @override
  Future<List<StyleInfo>> styles({String? category}) async => const [];
  @override
  String mediaUrl(String url) => 'http://fake$url';
  @override
  dynamic noSuchMethod(Invocation invocation) => throw UnimplementedError();
}

Future<AppState> pumpApp(
  WidgetTester t,
  FakeApi api, {
  required Map<String, Object> prefs,
}) async {
  SharedPreferences.setMockInitialValues(prefs);
  final state = await AppState.load(apiFactory: (_) => api);
  await t.pumpWidget(AxisApp(state: state));
  await t.pumpAndSettle();
  return state;
}

void main() {
  testWidgets('login then home', (t) async {
    final api = FakeApi();
    final state = await pumpApp(t, api, prefs: {});
    expect(find.byKey(const Key('loginButton')), findsOneWidget);
    await t.enterText(find.byKey(const Key('username')), 'hyun');
    await t.enterText(find.byKey(const Key('password')), 'secret1');
    await t.tap(find.byKey(const Key('loginButton')));
    await t.pumpAndSettle();
    expect(find.text('촬영 시작'), findsOneWidget);
    expect(api.token, 'tok-hyun');
    expect(state.username, 'hyun');
    expect(
      (await SharedPreferences.getInstance()).getString('token'),
      'tok-hyun',
    );
  });

  testWidgets('wrong password shows server message', (t) async {
    await pumpApp(t, FakeApi(), prefs: {});
    await t.enterText(find.byKey(const Key('username')), 'hyun');
    await t.enterText(find.byKey(const Key('password')), 'nope00');
    await t.tap(find.byKey(const Key('loginButton')));
    await t.pumpAndSettle();
    expect(find.text('아이디나 비밀번호가 틀려요'), findsOneWidget);
  });

  testWidgets('signup mode', (t) async {
    final api = FakeApi();
    await pumpApp(t, api, prefs: {});
    await t.tap(find.text('회원가입'));
    await t.pump();
    await t.enterText(find.byKey(const Key('username')), 'newbie');
    await t.enterText(find.byKey(const Key('password')), 'secret1');
    await t.tap(find.byKey(const Key('loginButton')));
    await t.pumpAndSettle();
    expect(api.signedUp, 'newbie');
    expect(find.text('촬영 시작'), findsOneWidget);
  });

  testWidgets('stored token skips login and is applied to api', (t) async {
    final api = FakeApi();
    await pumpApp(t, api, prefs: {'token': 'old', 'username': 'hyun'});
    expect(find.text('촬영 시작'), findsOneWidget);
    expect(api.token, 'old');
  });

  testWidgets('history_401_goes_to_login', (t) async {
    final api = FakeApi(shotsStatus: 401);
    final state = await pumpApp(
      t,
      api,
      prefs: {'token': 'old', 'username': 'hyun'},
    );
    await t.tap(find.text('촬영 기록'));
    await t.pumpAndSettle();
    expect(find.byKey(const Key('loginButton')), findsOneWidget);
    expect(state.token, isNull);
    expect(api.token, isNull);
  });
}
