import 'package:flutter/material.dart';

import '../api/models.dart';
import '../local/local_analyzer.dart';
import '../state/app_state.dart';
import 'artist_select_screen.dart';
import 'errors.dart';

const _icons = {'person': Icons.person, 'food': Icons.restaurant};

class CategoryScreen extends StatefulWidget {
  const CategoryScreen({super.key});

  @override
  State<CategoryScreen> createState() => _CategoryScreenState();
}

class _CategoryScreenState extends State<CategoryScreen> {
  Future<List<Category>>? _load;

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    // 폰 분석이거나 서버에 못 붙으면 앱에 넣은 목록을 쓴다.
    Future<List<Category>> local() =>
        LocalData.load().then((d) => d.categories);
    _load ??= state.analysis == 'local'
        ? local()
        : state.api.categories().catchError((Object _) => local());
    return Scaffold(
      appBar: AppBar(title: const Text('무엇을 찍을까요?')),
      body: FutureBuilder<List<Category>>(
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
              for (final c in snap.data!)
                Card(
                  child: ListTile(
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 16,
                    ),
                    leading: Icon(_icons[c.id] ?? Icons.photo_camera, size: 40),
                    title: Text(
                      c.name,
                      style: Theme.of(context).textTheme.headlineSmall,
                    ),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () {
                      state.selectCategory(c.id);
                      Navigator.of(context).push(
                        MaterialPageRoute<void>(
                          builder: (_) => ArtistSelectScreen(category: c),
                        ),
                      );
                    },
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}
