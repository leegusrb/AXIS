import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'models.dart';

class AxisApiException implements Exception {
  AxisApiException(this.status, this.body);
  final int status;
  final String body;

  @override
  String toString() => 'AxisApiException($status): $body';

  /// 서버 응답의 `detail` 문구(없거나 문구가 아니면 null).
  String? get detail {
    try {
      final d = (jsonDecode(body) as Map<String, dynamic>)['detail'];
      return d is String ? d : null;
    } catch (_) {
      return null;
    }
  }
}

/// 서버의 `/media/raw/...` 주소에 너비 제한(`?w=`)을 붙여 작은 사진을 받는다.
String withWidth(String url, int width) =>
    '$url${url.contains('?') ? '&' : '?'}w=$width';

abstract class AxisApi {
  /// 로그인 토큰. 있으면 모든 요청에 `Authorization: Bearer`로 붙인다.
  String? token;
  Future<Map<String, dynamic>> health();
  Future<AuthResult> signup(String username, String password);
  Future<AuthResult> login(String username, String password);
  Future<List<Category>> categories();
  Future<List<StyleInfo>> styles({String? category});
  Future<List<TastePhoto>> tastePhotos();
  Future<List<StyleScore>> recommend(List<String> likedPhotoIds);
  Future<GuideResult> analyze(
    Uint8List jpeg,
    String styleId, {
    String? scorer,
    bool depthPreview = false,
  });
  Future<Shot> saveShot({
    required Uint8List photo,
    required String styleId,
    required double score,
    required bool guideUsed,
  });
  Future<List<Shot>> shots({String? styleId});
  String mediaUrl(String url);
}

class HttpAxisApi implements AxisApi {
  HttpAxisApi(
    String baseUrl, {
    http.Client? client,
    Duration timeout = requestTimeout,
  }) : baseUrl = baseUrl.replaceAll(RegExp(r'/+$'), ''),
       _client = _TimeoutClient(client ?? http.Client(), timeout);

  /// 서버가 없을 때 촬영 저장·목록 요청이 오래 멈추지 않게 한다.
  static const requestTimeout = Duration(seconds: 5);

  final String baseUrl;
  final http.Client _client;
  @override
  String? token;

  Map<String, String> get _auth =>
      token == null ? const {} : {'authorization': 'Bearer $token'};

  static const analyzeTimeout = Duration(seconds: 3);

  Uri _uri(String path, [Map<String, String>? query]) =>
      Uri.parse('$baseUrl$path').replace(queryParameters: query);

  dynamic _decode(http.Response r) {
    if (r.statusCode < 200 || r.statusCode >= 300) {
      throw AxisApiException(r.statusCode, utf8.decode(r.bodyBytes));
    }
    return jsonDecode(utf8.decode(r.bodyBytes));
  }

  Future<dynamic> _multipart(
    String path,
    Map<String, String> fields,
    String fileField,
    Uint8List bytes,
  ) async {
    final req = http.MultipartRequest('POST', _uri(path))
      ..headers.addAll(_auth)
      ..fields.addAll(fields)
      ..files.add(
        http.MultipartFile.fromBytes(fileField, bytes, filename: 'shot.jpg'),
      );
    final resp = await http.Response.fromStream(await _client.send(req));
    return _decode(resp);
  }

  @override
  Future<Map<String, dynamic>> health() async =>
      _decode(await _client.get(_uri('/health'))) as Map<String, dynamic>;

  Future<AuthResult> _authPost(
    String path,
    String username,
    String password,
  ) async => AuthResult.fromJson(
    _decode(
      await _client.post(
        _uri(path),
        headers: {'content-type': 'application/json'},
        body: jsonEncode({'username': username, 'password': password}),
      ),
    ) as Map<String, dynamic>,
  );

  @override
  Future<AuthResult> signup(String username, String password) =>
      _authPost('/auth/signup', username, password);

  @override
  Future<AuthResult> login(String username, String password) =>
      _authPost('/auth/login', username, password);

  @override
  Future<List<Category>> categories() async =>
      (_decode(await _client.get(_uri('/categories'))) as List)
          .map((e) => Category.fromJson(e as Map<String, dynamic>))
          .toList();

  @override
  Future<List<StyleInfo>> styles({String? category}) async => (_decode(
    await _client.get(
      _uri('/styles', category == null ? null : {'category': category}),
    ),
  ) as List).map((e) => StyleInfo.fromJson(e as Map<String, dynamic>)).toList();

  @override
  Future<List<TastePhoto>> tastePhotos() async =>
      (_decode(await _client.get(_uri('/taste/photos'))) as List)
          .map((e) => TastePhoto.fromJson(e as Map<String, dynamic>))
          .toList();

  @override
  Future<List<StyleScore>> recommend(List<String> likedPhotoIds) async {
    final r = await _client.post(
      _uri('/recommend'),
      headers: {'content-type': 'application/json'},
      body: jsonEncode({'liked_photo_ids': likedPhotoIds}),
    );
    return (_decode(r) as List)
        .map((e) => StyleScore.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  @override
  Future<GuideResult> analyze(
    Uint8List jpeg,
    String styleId, {
    String? scorer,
    bool depthPreview = false,
  }) async {
    final j = await _multipart(
      '/analyze',
      {
        'style_id': styleId,
        'scorer': ?scorer,
        if (depthPreview) 'depth_preview': 'true',
      },
      'frame',
      jpeg,
    ).timeout(analyzeTimeout);
    return GuideResult.fromJson(j as Map<String, dynamic>);
  }

  @override
  Future<Shot> saveShot({
    required Uint8List photo,
    required String styleId,
    required double score,
    required bool guideUsed,
  }) async {
    final j = await _multipart(
      '/shots',
      {
        'style_id': styleId,
        'score': score.toString(),
        'guide_used': guideUsed ? 'true' : 'false',
      },
      'photo',
      photo,
    );
    return Shot.fromJson(j as Map<String, dynamic>);
  }

  @override
  Future<List<Shot>> shots({String? styleId}) async => (_decode(
    await _client.get(
      _uri('/shots', styleId == null ? null : {'style_id': styleId}),
      headers: _auth,
    ),
  ) as List).map((e) => Shot.fromJson(e as Map<String, dynamic>)).toList();

  @override
  String mediaUrl(String url) => url.startsWith('http')
      ? url
      : '$baseUrl${url.startsWith('/') ? '' : '/'}$url';
}

class _TimeoutClient extends http.BaseClient {
  _TimeoutClient(this._inner, this._timeout);
  final http.Client _inner;
  final Duration _timeout;

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) =>
      _inner.send(request).timeout(_timeout);

  @override
  void close() => _inner.close();
}
