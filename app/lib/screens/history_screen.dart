import 'package:flutter/material.dart';

import '../api/axis_api.dart';
import '../api/models.dart';
import '../state/app_state.dart';
import 'errors.dart';
import 'result_screen.dart';

String formatLocalDate(String iso) {
  final d = DateTime.tryParse(iso)?.toLocal();
  if (d == null) return iso;
  String t(int n) => n.toString().padLeft(2, '0');
  return '${d.year}.${t(d.month)}.${t(d.day)} ${t(d.hour)}:${t(d.minute)}';
}

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  Future<(List<Shot>, Map<String, String>)>? _load;
  String? _filter; // 작가 필터(null이면 전체)
  List<String> _chipIds = const []; // 필터 칩: 전체 기록에 있는 작가들

  Future<(List<Shot>, Map<String, String>)> _fetch(AppState state) async {
    final List<Shot> shots;
    try {
      shots = await state.api.shots(styleId: _filter);
    } on AxisApiException catch (e) {
      if (e.status == 401 && mounted) {
        // 토큰이 만료되었거나 서버가 모르는 토큰: 다시 로그인한다.
        await state.logout();
        if (mounted) Navigator.of(context).popUntil((r) => r.isFirst);
      }
      rethrow;
    }
    var names = <String, String>{};
    try {
      names = {for (final s in await state.api.styles()) s.id: s.name};
    } catch (_) {
      // 스타일 이름을 못 구해도 id로 표시한다.
    }
    if (_filter == null) {
      _chipIds = {for (final s in shots) s.styleId}.toList();
    }
    return (shots, names);
  }

  void _select(AppState state, String? id) {
    setState(() {
      _filter = id;
      _load = _fetch(state);
    });
  }

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    if (state.token == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('촬영 기록')),
        body: const Center(child: Text('서버에 연결하면 기록을 볼 수 있어요')),
      );
    }
    _load ??= _fetch(state);
    return Scaffold(
      appBar: AppBar(title: const Text('촬영 기록')),
      body: FutureBuilder<(List<Shot>, Map<String, String>)>(
        future: _load,
        builder: (context, snap) {
          if (snap.hasError) return const Center(child: Text(serverErrorText));
          if (!snap.hasData) {
            return const Center(child: CircularProgressIndicator());
          }
          final (shots, names) = snap.data!;
          if (shots.isEmpty && _filter == null) {
            return const Center(child: Text('아직 찍은 사진이 없어요'));
          }
          return ListView.builder(
            itemCount: shots.length + 1,
            itemBuilder: (context, i) {
              if (i == 0) {
                return Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 8,
                  ),
                  child: Wrap(
                    spacing: 8,
                    children: [
                      ChoiceChip(
                        label: const Text('전체'),
                        selected: _filter == null,
                        onSelected: (_) => _select(state, null),
                      ),
                      for (final id in _chipIds)
                        ChoiceChip(
                          label: Text(names[id] ?? id),
                          selected: _filter == id,
                          onSelected: (_) => _select(state, id),
                        ),
                    ],
                  ),
                );
              }
              final s = shots[i - 1];
              return ListTile(
                leading: SizedBox(
                  width: 56,
                  height: 56,
                  child: Image.network(
                    state.api.mediaUrl(s.url),
                    cacheWidth: 168, // 56dp 썸네일을 3배 화면에서 그릴 만큼만 디코드
                    fit: BoxFit.cover,
                    errorBuilder: (_, _, _) => const ColoredBox(
                      color: Colors.black12,
                      child: Icon(Icons.image_not_supported),
                    ),
                  ),
                ),
                title: Text(names[s.styleId] ?? s.styleId),
                subtitle: Text(
                  '${scorePoints(s.score)}점 · ${formatLocalDate(s.createdAt)}',
                ),
              );
            },
          );
        },
      ),
    );
  }
}
