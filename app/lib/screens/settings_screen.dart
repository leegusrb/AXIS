import 'package:flutter/material.dart';

import '../state/app_state.dart';
import 'errors.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

const _scorerOrder = ['stat', 'rank', 'head'];

String scorerLabel(String s) => switch (s) {
  'stat' => '통계',
  'rank' => '순위 학습',
  'head' => '통합 가이드 모델(Guide Head)',
  'rule' => '규칙',
  _ => s,
};

class _SettingsScreenState extends State<SettingsScreen> {
  TextEditingController? _controller;
  Future<List<String>>? _scorers;

  Future<List<String>> _loadScorers(AppState state) async {
    try {
      final h = await state.api.health();
      final have = ((h['scorers'] as List?) ?? const []).cast<String>().toSet();
      return [
        for (final s in _scorerOrder)
          if (have.contains(s)) s,
      ];
    } catch (_) {
      return const [];
    }
  }

  Future<void> _check(AppState state) async {
    await state.setServerUrl(_controller!.text);
    try {
      final h = await state.api.health();
      if (!mounted) return;
      final scorers = _loadScorers(state);
      setState(() {
        _scorers = scorers;
      });
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          SnackBar(content: Text('연결됨 (모델 ${h['model_version']})')),
        );
    } catch (_) {
      if (mounted) showServerError(context);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    _controller ??= TextEditingController(text: state.serverUrl);
    _scorers ??= _loadScorers(state);
    return Scaffold(
      appBar: AppBar(title: const Text('설정')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: ListView(
          children: [
            TextField(
              controller: _controller,
              keyboardType: TextInputType.url,
              textInputAction: TextInputAction.done,
              decoration: const InputDecoration(
                labelText: '서버 주소',
                hintText: 'http://192.168.0.2:8000',
                border: OutlineInputBorder(),
              ),
              onSubmitted: state.setServerUrl,
            ),
            const SizedBox(height: 12),
            FilledButton(
              onPressed: () => _check(state),
              child: const Text('연결 확인'),
            ),
            const SizedBox(height: 24),
            Text('분석 위치', style: Theme.of(context).textTheme.titleMedium),
            RadioGroup<String>(
              groupValue: state.analysis,
              onChanged: (v) {
                if (v != null) state.setAnalysis(v);
              },
              child: const Column(
                children: [
                  RadioListTile<String>(
                    value: 'server',
                    title: Text('서버에서 분석'),
                    subtitle: Text('깊이 보기·Guide Head·작가 추천·기록'),
                  ),
                  RadioListTile<String>(
                    value: 'local',
                    title: Text('폰에서 분석'),
                    subtitle: Text('서버 없이 인물·음식 구도와 수평 안내'),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            FutureBuilder<List<String>>(
              future: _scorers,
              builder: (context, snap) {
                final list = snap.data ?? const <String>[];
                if (list.isEmpty) return const SizedBox.shrink();
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '안내 방식',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    RadioGroup<String>(
                      groupValue: state.scorer,
                      onChanged: (v) {
                        if (v != null) state.setScorer(v);
                      },
                      child: Column(
                        children: [
                          for (final s in list)
                            RadioListTile<String>(
                              value: s,
                              title: Text(scorerLabel(s)),
                            ),
                        ],
                      ),
                    ),
                  ],
                );
              },
            ),
            if (state.guest && state.token == null) ...[
              const SizedBox(height: 24),
              const Text('서버 없이 쓰는 중이에요'),
              const SizedBox(height: 8),
              FilledButton.tonal(
                onPressed: () async {
                  await state.logout(); // 손님 모드를 끝내고 로그인 화면으로
                  if (context.mounted) {
                    Navigator.of(context).popUntil((r) => r.isFirst);
                  }
                },
                child: const Text('로그인하기'),
              ),
            ],
            if (state.token != null) ...[
              const SizedBox(height: 24),
              Text('로그인: ${state.username ?? ''}'),
              const SizedBox(height: 8),
              OutlinedButton(
                onPressed: () async {
                  await state.logout();
                  if (context.mounted) {
                    Navigator.of(context).popUntil((r) => r.isFirst);
                  }
                },
                child: const Text('로그아웃'),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
