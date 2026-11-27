import 'package:flutter/material.dart';

import '../api/axis_api.dart';
import '../api/models.dart';
import '../config.dart';
import '../local/local_analyzer.dart';
import '../state/app_state.dart';
import '../widgets/composition_diagram.dart';
import 'errors.dart';
import 'routes.dart';
import 'taste_screen.dart';

class ArtistSelectScreen extends StatefulWidget {
  const ArtistSelectScreen({super.key, required this.category});
  final Category category;

  @override
  State<ArtistSelectScreen> createState() => _ArtistSelectScreenState();
}

class _ArtistSelectScreenState extends State<ArtistSelectScreen> {
  Future<List<StyleInfo>>? _load;

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final cat = widget.category.id;
    Future<List<StyleInfo>> local() => LocalData.load().then(
      (d) => [
        for (final a in d.artists)
          if (a.category == cat) a,
      ],
    );
    _load ??= state.analysis == 'local'
        ? local()
        : state.api.styles(category: cat).catchError((Object _) => local());
    return Scaffold(
      appBar: AppBar(title: Text('${widget.category.name} 작가 고르기')),
      body: FutureBuilder<List<StyleInfo>>(
        future: _load,
        builder: (context, snap) {
          if (snap.hasError) {
            return const Center(child: Text(serverErrorText));
          }
          if (!snap.hasData) {
            return const Center(child: CircularProgressIndicator());
          }
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              // 작가 추천은 서버(MobileNet·취향 사진)가 필요하다.
              if (cat == 'person' &&
                  state.analysis == 'server' &&
                  state.token != null) ...[
                FilledButton.tonalIcon(
                  icon: const Icon(Icons.auto_awesome),
                  onPressed: () => Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => const TasteScreen(),
                    ),
                  ),
                  label: const Text('나에게 맞는 작가 찾기'),
                ),
                const SizedBox(height: 12),
              ],
              for (final s in snap.data!) _ArtistCard(style: s, api: state.api),
            ],
          );
        },
      ),
    );
  }
}

class _ArtistCard extends StatelessWidget {
  const _ArtistCard({required this.style, required this.api});
  final StyleInfo style;
  final AxisApi api;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final box = style.diagramBox;
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () {
          AppScope.of(context).selectStyle(style);
          openCamera(context);
        },
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(
                width: 72,
                child: box == null
                    ? const SizedBox(height: 96)
                    : CompositionDiagram(box: box),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(style.name, style: text.titleMedium),
                    const SizedBox(height: 4),
                    Text(style.description, style: text.bodySmall),
                    if (style.kind == 'rule') ...[
                      const SizedBox(height: 6),
                      const Chip(
                        label: Text('규칙 기반'),
                        visualDensity: VisualDensity.compact,
                      ),
                    ],
                    if (style.samples.isNotEmpty) ...[
                      const SizedBox(height: 8),
                      SizedBox(
                        height: 64,
                        child: Row(
                          children: [
                            for (final p in style.samples.take(3))
                              Padding(
                                padding: const EdgeInsets.only(right: 6),
                                child: SizedBox(
                                  width: 48,
                                  child: Image.network(
                                    api.mediaUrl(
                                      withWidth(p.url, mediaThumbWidth),
                                    ),
                                    cacheWidth: mediaThumbWidth,
                                    fit: BoxFit.cover,
                                    errorBuilder: (_, _, _) =>
                                        const ColoredBox(color: Colors.black12),
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ),
                      Text(
                        '사진: ${style.samples.map((p) => '${p.author} · ${p.license}').toSet().join(', ')}',
                        style: text.bodySmall,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
