import 'package:flutter/material.dart';

import '../api/axis_api.dart';
import '../api/models.dart';
import '../config.dart';
import '../state/app_state.dart';
import 'errors.dart';
import 'routes.dart';

class RecommendScreen extends StatefulWidget {
  const RecommendScreen({super.key});

  @override
  State<RecommendScreen> createState() => _RecommendScreenState();
}

class _RecommendScreenState extends State<RecommendScreen> {
  Future<List<StyleInfo>>? _styles;
  String? _currentId;

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    _styles ??= state.api.styles();
    final scores = state.lastRecommendation;
    return Scaffold(
      appBar: AppBar(title: const Text('추천')),
      body: FutureBuilder<List<StyleInfo>>(
        future: _styles,
        builder: (context, snap) {
          if (snap.hasError) return const Center(child: Text(serverErrorText));
          if (!snap.hasData) {
            return const Center(child: CircularProgressIndicator());
          }
          final styles = {for (final s in snap.data!) s.id: s};
          final known = scores
              .where((s) => styles.containsKey(s.styleId))
              .toList();
          if (known.isEmpty) return const Center(child: Text(serverErrorText));
          final current = known.firstWhere(
            (s) => s.styleId == _currentId,
            orElse: () => known.first,
          );
          final style = styles[current.styleId]!;
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Text(
                "당신에게 맞는 작가: '${style.name}' (${current.score.round()}%)",
                style: Theme.of(context).textTheme.titleLarge,
              ),
              const SizedBox(height: 16),
              SizedBox(
                height: 200,
                child: ListView(
                  scrollDirection: Axis.horizontal,
                  children: [
                    for (final s in style.samples.take(3))
                      Padding(
                        padding: const EdgeInsets.only(right: 8),
                        child: SizedBox(
                          width: 130,
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Expanded(
                                child: SizedBox.expand(
                                  child: Image.network(
                                    state.api.mediaUrl(
                                      withWidth(s.url, mediaThumbWidth),
                                    ),
                                    cacheWidth: mediaThumbWidth,
                                    fit: BoxFit.cover,
                                    errorBuilder: (_, _, _) => const ColoredBox(
                                      color: Colors.black12,
                                      child: Icon(Icons.image_not_supported),
                                    ),
                                  ),
                                ),
                              ),
                              Text(
                                '사진: ${s.author} · ${s.license}',
                                style: Theme.of(context).textTheme.bodySmall,
                              ),
                            ],
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              Wrap(
                spacing: 8,
                children: [
                  for (final s in known)
                    ChoiceChip(
                      label: Text(styles[s.styleId]!.name),
                      selected: s.styleId == current.styleId,
                      onSelected: (_) {
                        setState(() => _currentId = s.styleId);
                        state.selectStyle(styles[s.styleId]);
                      },
                    ),
                ],
              ),
              const SizedBox(height: 24),
              FilledButton(
                onPressed: () {
                  state.selectStyle(style);
                  openCamera(context);
                },
                child: const Text('이 작가로 촬영'),
              ),
            ],
          );
        },
      ),
    );
  }
}
