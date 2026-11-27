// ai/axis_ai/scorers/stat.py, rule.py를 옮긴 것.
import 'dart:math' as math;

import 'geometry.dart';

abstract class LocalScorer {
  double score(Features f);
}

class StatScorer implements LocalScorer {
  StatScorer(this.means, this.invCovs);
  final List<List<double>> means; // K×6
  final List<List<List<double>>> invCovs; // K×6×6

  @override
  double score(Features f) {
    final v = f.vector;
    var best = double.infinity;
    for (var k = 0; k < means.length; k++) {
      final d = [for (var i = 0; i < 6; i++) v[i] - means[k][i]];
      var d2 = 0.0;
      for (var i = 0; i < 6; i++) {
        for (var j = 0; j < 6; j++) {
          d2 += d[i] * invCovs[k][i][j] * d[j];
        }
      }
      best = math.min(best, d2);
    }
    return math.exp(-best / 12).clamp(0.0, 1.0);
  }
}

class RuleScorer implements LocalScorer {
  RuleScorer(this.target, this.spread);
  final Map<String, double> target, spread;

  @override
  double score(Features f) {
    var z2 = 0.0;
    for (final e in target.entries) {
      final z = (f.get(e.key) - e.value) / spread[e.key]!;
      z2 += z * z;
    }
    return math.exp(-0.5 * z2);
  }
}

class AnyOfScorer implements LocalScorer {
  AnyOfScorer(this.parts);
  final List<RuleScorer> parts;

  @override
  double score(Features f) => parts.map((p) => p.score(f)).reduce(math.max);
}

List<double> _vec(Object? v) => [
  for (final x in v as List) (x as num).toDouble(),
];
Map<String, double> _map(Object? v) => (v as Map<String, dynamic>).map(
  (k, x) => MapEntry(k, (x as num).toDouble()),
);

/// local.json의 stat(학습 작가)과 rules(규칙 작가)를 작가 id별 점수기로 만든다.
Map<String, LocalScorer> loadScorers(Map<String, dynamic> local) => {
  for (final e in (local['stat'] as Map<String, dynamic>).entries)
    e.key: StatScorer(
      [for (final m in (e.value as Map)['means'] as List) _vec(m)],
      [
        for (final c in (e.value as Map)['inv_covs'] as List)
          [for (final r in c as List) _vec(r)],
      ],
    ),
  for (final e in (local['rules'] as Map<String, dynamic>).entries)
    e.key: () {
      final parts = [
        for (final r in e.value as List)
          RuleScorer(_map(r['target']), _map(r['spread'])),
      ];
      return parts.length == 1
          ? parts.single
          : AnyOfScorer(parts) as LocalScorer;
    }(),
};
