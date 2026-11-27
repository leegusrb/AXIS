import 'dart:ui' show Rect;

double _d(Object? v) => (v as num).toDouble();

Rect? _box(Object? v) {
  if (v == null) return null;
  final l = (v as List).map(_d).toList();
  return Rect.fromLTWH(l[0], l[1], l[2], l[3]);
}

class SubjectInfo {
  const SubjectInfo({
    required this.className,
    required this.cx,
    required this.cy,
    required this.w,
    required this.h,
    required this.area,
    required this.conf,
  });
  final String className;
  final double cx, cy, w, h, area, conf;

  static SubjectInfo? fromJson(Object? v) {
    if (v == null) return null;
    final j = v as Map<String, dynamic>;
    return SubjectInfo(
      className: j['class_name'] as String,
      cx: _d(j['cx']),
      cy: _d(j['cy']),
      w: _d(j['w']),
      h: _d(j['h']),
      area: _d(j['area']),
      conf: _d(j['conf']),
    );
  }
}

class GuideResult {
  const GuideResult({
    this.category = 'person',
    this.subject,
    this.rotationDeg,
    this.rotate,
    this.depthPreview,
    this.depthDiff,
    required this.direction,
    required this.magnitude,
    required this.message,
    required this.score,
    required this.scoreDelta,
    required this.confidence,
    required this.subjectBox,
    required this.targetBox,
    required this.scorer,
    required this.modelVersion,
    required this.timingMs,
  });

  final String direction;
  final String magnitude;
  final String message;
  final double score;
  final double scoreDelta;
  final double confidence;
  final Rect? subjectBox;
  final Rect? targetBox;
  final String scorer;
  final String modelVersion;
  final Map<String, int> timingMs;
  final String category;
  final SubjectInfo? subject;
  final double? rotationDeg;
  final String? rotate; // 휴대폰을 돌릴 방향 CW/CCW, 필요 없으면 null
  final String? depthPreview; // base64 PNG(깊이 보기를 켰을 때만)
  final double? depthDiff; // 피사체 깊이 - 배경 깊이

  factory GuideResult.fromJson(Map<String, dynamic> j) => GuideResult(
    category: j['category'] as String? ?? 'person',
    subject: SubjectInfo.fromJson(j['subject']),
    rotationDeg: j['rotation_deg'] == null ? null : _d(j['rotation_deg']),
    rotate: j['rotate'] as String?,
    depthPreview: j['depth_preview'] as String?,
    depthDiff: j['depth'] == null
        ? null
        : _d((j['depth'] as Map<String, dynamic>)['diff']),
    direction: j['direction'] as String,
    magnitude: j['magnitude'] as String,
    message: j['message'] as String,
    score: _d(j['score']),
    scoreDelta: _d(j['score_delta']),
    confidence: _d(j['confidence']),
    subjectBox: _box(j['subject_box']),
    targetBox: _box(j['target_box']),
    scorer: j['scorer'] as String,
    modelVersion: j['model_version'] as String,
    timingMs: (j['timing_ms'] as Map<String, dynamic>).map(
      (k, v) => MapEntry(k, (v as num).toInt()),
    ),
  );
}

class StyleSample {
  const StyleSample({
    required this.url,
    required this.author,
    required this.license,
  });
  final String url;
  final String author;
  final String license;

  factory StyleSample.fromJson(Map<String, dynamic> j) => StyleSample(
    url: j['url'] as String,
    author: j['author'] as String,
    license: j['license'] as String,
  );
}

class StyleInfo {
  const StyleInfo({
    required this.id,
    required this.name,
    required this.description,
    required this.samples,
    this.category = 'person',
    this.kind = 'learned',
    this.modelVersion = '',
    this.diagramBox,
  });
  final String id;
  final String name;
  final String description;
  final List<StyleSample> samples;
  final String category;
  final String kind; // learned | rule
  final String modelVersion;
  final Rect? diagramBox;

  factory StyleInfo.fromJson(Map<String, dynamic> j) => StyleInfo(
    id: j['id'] as String,
    name: j['name'] as String,
    description: j['description'] as String,
    samples: (j['samples'] as List)
        .map((e) => StyleSample.fromJson(e as Map<String, dynamic>))
        .toList(),
    category: j['category'] as String? ?? 'person',
    kind: j['kind'] as String? ?? 'learned',
    modelVersion: j['model_version'] as String? ?? '',
    diagramBox: _box(j['diagram_box']),
  );
}

class Category {
  const Category({required this.id, required this.name});
  final String id;
  final String name;

  factory Category.fromJson(Map<String, dynamic> j) =>
      Category(id: j['id'] as String, name: j['name'] as String);
}

class AuthResult {
  const AuthResult({required this.token, required this.username});
  final String token;
  final String username;

  factory AuthResult.fromJson(Map<String, dynamic> j) => AuthResult(
    token: j['token'] as String,
    username: j['username'] as String,
  );
}

class TastePhoto {
  const TastePhoto({required this.id, required this.url});
  final String id;
  final String url;

  factory TastePhoto.fromJson(Map<String, dynamic> j) =>
      TastePhoto(id: j['id'] as String, url: j['url'] as String);
}

class StyleScore {
  const StyleScore({required this.styleId, required this.score});
  final String styleId;
  final double score;

  factory StyleScore.fromJson(Map<String, dynamic> j) =>
      StyleScore(styleId: j['style_id'] as String, score: _d(j['score']));
}

class Shot {
  const Shot({
    required this.id,
    required this.styleId,
    required this.score,
    required this.guideUsed,
    required this.url,
    required this.createdAt,
  });
  final String id;
  final String styleId;
  final double score;
  final bool guideUsed;
  final String url;
  final String createdAt;

  factory Shot.fromJson(Map<String, dynamic> j) => Shot(
    id: j['id'] as String,
    styleId: j['style_id'] as String,
    score: _d(j['score']),
    guideUsed: j['guide_used'] as bool,
    url: j['url'] as String,
    createdAt: j['created_at'] as String,
  );
}
