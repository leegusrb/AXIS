import 'package:flutter/material.dart';

import 'routes.dart';
import 'settings_screen.dart';
import 'category_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AXIS'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: '설정',
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute<void>(builder: (_) => const SettingsScreen()),
            ),
          ),
        ],
      ),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            FilledButton(
              onPressed: () => Navigator.of(context).push(
                MaterialPageRoute<void>(builder: (_) => const CategoryScreen()),
              ),
              child: const Text('촬영 시작'),
            ),
            const SizedBox(height: 12),
            OutlinedButton(
              onPressed: () => openHistory(context),
              child: const Text('촬영 기록'),
            ),
          ],
        ),
      ),
    );
  }
}
