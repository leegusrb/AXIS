import 'package:flutter/material.dart';

import '../api/axis_api.dart';
import '../api/models.dart';
import '../config.dart';
import '../state/app_state.dart';
import 'errors.dart';
import 'recommend_screen.dart';

class TasteScreen extends StatefulWidget {
  const TasteScreen({super.key});

  @override
  State<TasteScreen> createState() => _TasteScreenState();
}

class _TasteScreenState extends State<TasteScreen> {
  Future<List<TastePhoto>>? _photos;
  final _selected = <String>{};
  bool _loading = false;

  Future<void> _recommend(AppState state) async {
    setState(() => _loading = true);
    try {
      final scores = await state.api.recommend(_selected.toList());
      state.setRecommendation(scores);
      if (!mounted) return;
      await Navigator.of(
        context,
      ).push(MaterialPageRoute<void>(builder: (_) => const RecommendScreen()));
    } catch (_) {
      if (mounted) showServerError(context);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    _photos ??= state.api.tastePhotos();
    return Scaffold(
      appBar: AppBar(title: const Text('마음에 드는 사진을 골라주세요')),
      body: FutureBuilder<List<TastePhoto>>(
        future: _photos,
        builder: (context, snap) {
          if (snap.hasError) {
            return Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text(serverErrorText),
                  TextButton(
                    onPressed: () =>
                        setState(() => _photos = state.api.tastePhotos()),
                    child: const Text('다시 시도'),
                  ),
                ],
              ),
            );
          }
          if (!snap.hasData) {
            return const Center(child: CircularProgressIndicator());
          }
          return Column(
            children: [
              Expanded(
                child: GridView.count(
                  crossAxisCount: 3,
                  mainAxisSpacing: 4,
                  crossAxisSpacing: 4,
                  padding: const EdgeInsets.all(4),
                  children: [for (final p in snap.data!) _tile(state, p)],
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(16),
                child: SizedBox(
                  width: double.infinity,
                  child: FilledButton(
                    onPressed: _selected.isEmpty || _loading
                        ? null
                        : () => _recommend(state),
                    child: _loading
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Text('추천 받기'),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _tile(AppState state, TastePhoto p) {
    final on = _selected.contains(p.id);
    return GestureDetector(
      key: ValueKey('taste-${p.id}'),
      onTap: () =>
          setState(() => on ? _selected.remove(p.id) : _selected.add(p.id)),
      child: Stack(
        fit: StackFit.expand,
        children: [
          Image.network(
            state.api.mediaUrl(withWidth(p.url, mediaThumbWidth)),
            cacheWidth: mediaThumbWidth,
            fit: BoxFit.cover,
            errorBuilder: (_, _, _) => const ColoredBox(
              color: Colors.black12,
              child: Icon(Icons.image_not_supported),
            ),
          ),
          if (on)
            const ColoredBox(
              color: Color(0x66000000),
              child: Align(
                alignment: Alignment.topRight,
                child: Padding(
                  padding: EdgeInsets.all(4),
                  child: Icon(Icons.check_circle, color: Colors.white),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
