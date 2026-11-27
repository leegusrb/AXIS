import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui';

import 'package:axis_app/api/axis_api.dart';
import 'package:axis_app/api/models.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

const _json = '''
{
  "direction": "LEFT",
  "magnitude": "SMALL",
  "message": "카메라를 왼쪽으로 조금 옮기세요",
  "score": 0.54,
  "score_delta": 0.07,
  "confidence": 0.82,
  "subject_box": [0.31, 0.12, 0.28, 0.71],
  "target_box": [0.40, 0.10, 0.28, 0.71],
  "scorer": "stat",
  "model_version": "stat-2026-11-22",
  "timing_ms": {"decode": 4, "pose": 38, "guide": 2, "total": 46},
  "category": "person",
  "subject": {"class_name": "person", "cx": 0.45, "cy": 0.475, "w": 0.28, "h": 0.71, "area": 0.2, "conf": 0.9},
  "rotation_deg": 10.5,
  "rotate": "CW",
  "depth": {"subject": 0.8, "background": 0.3, "diff": 0.5},
  "depth_preview": "abc="
}
''';

void main() {
  test('GuideResult parses server json', () {
    final r = GuideResult.fromJson(jsonDecode(_json) as Map<String, dynamic>);
    expect(r.direction, 'LEFT');
    expect(r.magnitude, 'SMALL');
    expect(r.message, '카메라를 왼쪽으로 조금 옮기세요');
    expect(r.score, 0.54);
    expect(r.scoreDelta, 0.07);
    expect(r.confidence, 0.82);
    expect(r.subjectBox, const Rect.fromLTWH(0.31, 0.12, 0.28, 0.71));
    expect(r.targetBox, const Rect.fromLTWH(0.40, 0.10, 0.28, 0.71));
    expect(r.scorer, 'stat');
    expect(r.modelVersion, 'stat-2026-11-22');
    expect(r.timingMs, {'decode': 4, 'pose': 38, 'guide': 2, 'total': 46});
    expect(r.rotate, 'CW');
    expect(r.depthPreview, 'abc=');
    expect(r.depthDiff, 0.5);
    expect(r.rotationDeg, 10.5);
    expect(r.category, 'person');
    expect(r.subject!.className, 'person');
    expect(r.subject!.conf, 0.9);
    final old = GuideResult.fromJson(
      (jsonDecode(_json) as Map<String, dynamic>)
        ..remove('rotate')
        ..remove('rotation_deg')
        ..remove('subject')
        ..remove('category'),
    );
    expect(old.rotate, isNull);
    expect(old.subject, isNull);
    expect(old.category, 'person');
  });

  test('GuideResult tolerates ints and null boxes', () {
    final j = jsonDecode(_json) as Map<String, dynamic>;
    j['score'] = 1;
    j['confidence'] = 0;
    j['subject_box'] = null;
    j['target_box'] = [0, 0, 1, 1];
    final r = GuideResult.fromJson(j);
    expect(r.score, 1.0);
    expect(r.confidence, 0.0);
    expect(r.subjectBox, isNull);
    expect(r.targetBox, const Rect.fromLTWH(0, 0, 1, 1));
  });

  test('other models parse', () {
    final s = StyleInfo.fromJson({
      'id': 's1',
      'name': 'n',
      'description': 'd',
      'samples': [
        {'url': '/media/raw/a.jpg', 'author': 'x', 'license': 'CC0'},
      ],
    });
    expect(s.samples.single.url, '/media/raw/a.jpg');
    expect(StyleScore.fromJson({'style_id': 's1', 'score': 1}).score, 1.0);
    expect(TastePhoto.fromJson({'id': 'p', 'url': '/u'}).id, 'p');
    final shot = Shot.fromJson({
      'id': '1',
      'style_id': 's1',
      'score': 0.5,
      'guide_used': true,
      'url': '/media/shots/1.jpg',
      'created_at': '2026-11-16T00:00:00',
    });
    expect(shot.guideUsed, true);
  });

  group('HttpAxisApi', () {
    test('mediaUrl joins base url', () {
      final api = HttpAxisApi('http://h:8000/');
      expect(api.mediaUrl('/media/raw/a.jpg'), 'http://h:8000/media/raw/a.jpg');
      expect(api.mediaUrl('media/a.jpg'), 'http://h:8000/media/a.jpg');
    });

    test('analyze sends multipart frame and style_id', () async {
      late http.BaseRequest seen;
      late String body;
      final client = MockClient.streaming((req, bodyStream) async {
        seen = req;
        body = latin1.decode(await bodyStream.toBytes());
        return http.StreamedResponse(
          Stream.value(utf8.encode(_json)),
          200,
          headers: {'content-type': 'application/json'},
        );
      });
      final api = HttpAxisApi('http://h:8000', client: client);
      final r = await api.analyze(Uint8List.fromList([1, 2, 3]), 'wide');
      expect(r.direction, 'LEFT');
      expect(seen.url.toString(), 'http://h:8000/analyze');
      expect(body, contains('name="style_id"'));
      expect(body, contains('wide'));
      expect(body, contains('name="frame"'));
    });

    test('saveShot sends fields and filename', () async {
      late String body;
      final client = MockClient.streaming((req, bodyStream) async {
        body = latin1.decode(await bodyStream.toBytes());
        return http.StreamedResponse(
          Stream.value(
            utf8.encode(
              jsonEncode({
                'id': '1',
                'style_id': 's',
                'score': 0.5,
                'guide_used': true,
                'url': '/media/shots/1.jpg',
                'created_at': 't',
              }),
            ),
          ),
          200,
        );
      });
      final api = HttpAxisApi('http://h', client: client);
      final s = await api.saveShot(
        photo: Uint8List.fromList([1]),
        styleId: 's',
        score: 0.5,
        guideUsed: true,
      );
      expect(s.url, '/media/shots/1.jpg');
      expect(body, contains('name="photo"'));
      expect(body, contains('filename="shot.jpg"'));
      expect(body, isNot(contains('name="device_id"')));
      expect(body, contains('name="score"'));
      expect(body, contains('name="guide_used"'));
      expect(body, contains('true'));
    });

    test('non-2xx throws AxisApiException', () async {
      final client = MockClient((req) async => http.Response('nope', 404));
      final api = HttpAxisApi('http://h', client: client);
      expect(
        api.styles(),
        throwsA(
          isA<AxisApiException>()
              .having((e) => e.status, 'status', 404)
              .having((e) => e.body, 'body', 'nope'),
        ),
      );
    });

    test('shots sends bearer token and style filter', () async {
      late http.Request seen;
      final client = MockClient((req) async {
        seen = req;
        return http.Response('[]', 200);
      });
      final api = HttpAxisApi('http://h', client: client)..token = 'tk';
      expect(await api.shots(styleId: 'wide_full'), isEmpty);
      expect(seen.url.queryParameters['style_id'], 'wide_full');
      expect(seen.headers['authorization'], 'Bearer tk');
      await api.shots();
      expect(seen.url.queryParameters, isEmpty);
    });

    test('login posts json and parses token', () async {
      late http.Request seen;
      final client = MockClient((req) async {
        seen = req;
        return http.Response('{"token":"t1","username":"hyun"}', 200);
      });
      final r = await HttpAxisApi(
        'http://h',
        client: client,
      ).login('hyun', 'pw1234');
      expect(r.token, 't1');
      expect(seen.url.path, '/auth/login');
      expect(jsonDecode(seen.body), {'username': 'hyun', 'password': 'pw1234'});
    });

    test('analyze sends scorer and depth flag only when given', () async {
      final bodies = <String>[];
      final client = MockClient.streaming((req, s) async {
        bodies.add(latin1.decode(await s.toBytes()));
        return http.StreamedResponse(Stream.value(utf8.encode(_json)), 200);
      });
      final api = HttpAxisApi('http://h', client: client);
      await api.analyze(Uint8List.fromList([1]), 'w');
      await api.analyze(
        Uint8List.fromList([1]),
        'w',
        scorer: 'head',
        depthPreview: true,
      );
      expect(bodies[0], isNot(contains('name="scorer"')));
      expect(bodies[0], isNot(contains('name="depth_preview"')));
      expect(bodies[1], contains('name="scorer"'));
      expect(bodies[1], contains('name="depth_preview"'));
    });

    test('styles passes category and parses new fields', () async {
      late Uri url;
      final client = MockClient((req) async {
        url = req.url;
        return http.Response(
          jsonEncode([
            {
              'id': 'food_center',
              'name': '작가 E',
              'description': '',
              'samples': [],
              'category': 'food',
              'kind': 'rule',
              'model_version': 'rule-v0',
              'diagram_box': [0.2, 0.2, 0.6, 0.6],
            },
          ]),
          200,
          headers: {'content-type': 'application/json; charset=utf-8'},
        );
      });
      final s = (await HttpAxisApi(
        'http://h',
        client: client,
      ).styles(category: 'food')).single;
      expect(url.queryParameters['category'], 'food');
      expect(s.kind, 'rule');
      expect(s.diagramBox!.width, closeTo(0.6, 1e-9));
    });

    test(
      'saveShot and lists time out when the server does not answer',
      () async {
        final client = MockClient.streaming(
          (req, s) => Completer<http.StreamedResponse>().future,
        );
        final api = HttpAxisApi(
          'http://h',
          client: client,
          timeout: const Duration(milliseconds: 20),
        );
        await expectLater(
          api.saveShot(
            photo: Uint8List(1),
            styleId: 's',
            score: 0,
            guideUsed: true,
          ),
          throwsA(isA<TimeoutException>()),
        );
        await expectLater(api.categories(), throwsA(isA<TimeoutException>()));
        await expectLater(api.styles(), throwsA(isA<TimeoutException>()));
      },
    );

    test('AxisApiException.detail', () {
      expect(
        AxisApiException(409, '{"detail":"이미 있는 아이디예요"}').detail,
        '이미 있는 아이디예요',
      );
      expect(AxisApiException(422, '{"detail":[{"loc":1}]}').detail, isNull);
      expect(AxisApiException(500, 'boom').detail, isNull);
    });
  });

  test('withWidth appends the w query', () {
    expect(withWidth('/media/raw/a.jpg', 600), '/media/raw/a.jpg?w=600');
    expect(
      withWidth('/media/raw/a.jpg?x=1', 600),
      '/media/raw/a.jpg?x=1&w=600',
    );
  });
}
