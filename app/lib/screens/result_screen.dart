import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../api/models.dart';
import '../state/app_state.dart';

/// 0–1 점수를 0–100 정수로 바꾼다.
int scorePoints(double score) => (score * 100).round().clamp(0, 100);

class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key, required this.shot, required this.photo});
  final Shot shot;
  final Uint8List photo;

  @override
  Widget build(BuildContext context) {
    final style = AppScope.of(context).selectedStyle;
    final styleName = style?.id == shot.styleId ? style!.name : shot.styleId;
    return Scaffold(
      appBar: AppBar(title: const Text('촬영 결과')),
      body: Column(
        children: [
          Expanded(
            child: Image.memory(
              photo,
              fit: BoxFit.contain,
              width: double.infinity,
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                Text(styleName, style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 4),
                Text(
                  '구도 점수 ${scorePoints(shot.score)}점',
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                const SizedBox(height: 12),
                FilledButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('계속 촬영'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
