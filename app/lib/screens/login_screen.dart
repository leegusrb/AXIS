import 'package:flutter/material.dart';

import '../api/axis_api.dart';
import '../state/app_state.dart';
import 'errors.dart';
import 'settings_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _user = TextEditingController();
  final _pw = TextEditingController();
  bool _signup = false;
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    _user.dispose();
    _pw.dispose();
    super.dispose();
  }

  Future<void> _submit(AppState state) async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final u = _user.text.trim();
      final r = _signup
          ? await state.api.signup(u, _pw.text)
          : await state.api.login(u, _pw.text);
      await state.setAuth(r);
    } on AxisApiException catch (e) {
      setState(
        () => _error =
            e.detail ??
            (e.status == 422
                ? '아이디는 3–20자(한글·영문·숫자·_), 비밀번호는 6자 이상이에요'
                : serverErrorText),
      );
    } catch (_) {
      setState(() => _error = serverErrorText);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    return Scaffold(
      appBar: AppBar(
        title: Text(_signup ? '회원가입' : '로그인'),
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
      body: ListView(
        padding: const EdgeInsets.all(24),
        children: [
          const Text(
            'AXIS',
            style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 24),
          TextField(
            key: const Key('username'),
            controller: _user,
            decoration: const InputDecoration(
              labelText: '아이디',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            key: const Key('password'),
            controller: _pw,
            obscureText: true,
            decoration: const InputDecoration(
              labelText: '비밀번호',
              border: OutlineInputBorder(),
            ),
            onSubmitted: (_) => _submit(state),
          ),
          if (_error != null) ...[
            const SizedBox(height: 12),
            Text(
              _error!,
              style: TextStyle(color: Theme.of(context).colorScheme.error),
            ),
          ],
          const SizedBox(height: 16),
          FilledButton(
            key: const Key('loginButton'),
            onPressed: _busy ? null : () => _submit(state),
            child: Text(_signup ? '가입하고 시작' : '로그인'),
          ),
          TextButton(
            onPressed: () => setState(() {
              _signup = !_signup;
              _error = null;
            }),
            child: Text(_signup ? '이미 계정이 있어요' : '회원가입'),
          ),
          const Divider(height: 32),
          OutlinedButton.icon(
            key: const Key('guestButton'),
            icon: const Icon(Icons.phone_iphone),
            onPressed: state.startGuest,
            label: const Text('서버 없이 시작(폰에서 분석)'),
          ),
        ],
      ),
    );
  }
}
