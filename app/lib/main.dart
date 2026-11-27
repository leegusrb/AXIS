import 'package:flutter/material.dart';

import 'screens/home_screen.dart';
import 'screens/login_screen.dart';
import 'state/app_state.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final state = await AppState.load();
  runApp(AxisApp(state: state));
}

class AxisApp extends StatelessWidget {
  const AxisApp({super.key, required this.state});
  final AppState state;

  @override
  Widget build(BuildContext context) {
    return AppScope(
      notifier: state,
      child: MaterialApp(
        title: 'AXIS',
        theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.indigo),
        home: const AuthGate(),
      ),
    );
  }
}

/// 토큰이 없으면 로그인 화면, 있으면 홈. 로그아웃하면 다시 로그인 화면이 된다.
class AuthGate extends StatelessWidget {
  const AuthGate({super.key});

  @override
  Widget build(BuildContext context) =>
      AppScope.of(context).signedIn ? const HomeScreen() : const LoginScreen();
}
