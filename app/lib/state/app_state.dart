import 'package:flutter/widgets.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';

import '../api/axis_api.dart';
import '../api/models.dart';

const defaultServerUrl = 'http://192.168.0.2:8000';

class AppState extends ChangeNotifier {
  AppState._(this._prefs, this._serverUrl, this.deviceId, this._apiFactory)
    : _api = _apiFactory(_serverUrl) {
    token = _prefs.getString('token');
    username = _prefs.getString('username');
    scorer = _prefs.getString('scorer') ?? 'stat';
    analysis = _prefs.getString('analysis') ?? 'server';
    guest = _prefs.getBool('guest') ?? false;
    _api.token = token;
  }

  final SharedPreferences _prefs;
  final AxisApi Function(String baseUrl) _apiFactory;
  String _serverUrl;
  AxisApi _api;
  final String deviceId;

  StyleInfo? selectedStyle;
  String? selectedCategory;
  List<StyleScore> lastRecommendation = const [];
  String? token;
  String? username;
  late String scorer;
  late String analysis; // 'server' | 'local'(폰에서 분석)
  late bool guest; // 서버 없이 시작

  bool get signedIn => token != null || guest;

  static Future<AppState> load({
    SharedPreferences? prefs,
    AxisApi Function(String baseUrl)? apiFactory,
  }) async {
    final p = prefs ?? await SharedPreferences.getInstance();
    var id = p.getString('deviceId');
    if (id == null) {
      id = const Uuid().v4();
      await p.setString('deviceId', id);
    }
    return AppState._(
      p,
      p.getString('serverUrl') ?? defaultServerUrl,
      id,
      apiFactory ?? (url) => HttpAxisApi(url),
    );
  }

  String get serverUrl => _serverUrl;
  AxisApi get api => _api;

  Future<void> setServerUrl(String url) async {
    _serverUrl = url.trim();
    _api = _apiFactory(_serverUrl)..token = token;
    await _prefs.setString('serverUrl', _serverUrl);
    notifyListeners();
  }

  Future<void> setAuth(AuthResult r) async {
    token = r.token;
    username = r.username;
    _api.token = token;
    await _prefs.setString('token', r.token);
    await _prefs.setString('username', r.username);
    notifyListeners();
  }

  Future<void> startGuest() async {
    guest = true;
    await _prefs.setBool('guest', true);
    await setAnalysis('local');
  }

  Future<void> setAnalysis(String a) async {
    analysis = a;
    await _prefs.setString('analysis', a);
    notifyListeners();
  }

  Future<void> logout() async {
    token = null;
    username = null;
    guest = false;
    await _prefs.remove('guest');
    _api.token = null;
    await _prefs.remove('token');
    await _prefs.remove('username');
    notifyListeners();
  }

  Future<void> setScorer(String s) async {
    scorer = s;
    await _prefs.setString('scorer', s);
    notifyListeners();
  }

  void selectCategory(String id) {
    selectedCategory = id;
    notifyListeners();
  }

  void selectStyle(StyleInfo? style) {
    selectedStyle = style;
    notifyListeners();
  }

  void setRecommendation(List<StyleScore> scores) {
    lastRecommendation = scores;
    notifyListeners();
  }
}

class AppScope extends InheritedNotifier<AppState> {
  const AppScope({
    super.key,
    required AppState super.notifier,
    required super.child,
  });

  static AppState of(BuildContext context) =>
      context.dependOnInheritedWidgetOfExactType<AppScope>()!.notifier!;
}
