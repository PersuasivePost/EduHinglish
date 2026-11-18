import 'package:flutter/material.dart';
import 'package:flutter_web_plugins/url_strategy.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'auth.dart';
import 'state.dart';
import 'theme.dart';
import 'ui/home_page.dart';
import 'config.dart';
import 'ui/login_page.dart';
import 'ui/shared_chat_page.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  usePathUrlStrategy(); // clean /s/<token> addresses on the web (no #)
  final prefs = await SharedPreferences.getInstance();
  runApp(ProviderScope(
    retry: (_, _) => null, // show errors right away instead of silently retrying
    overrides: [sharedPrefsProvider.overrideWithValue(prefs)],
    child: const EduHinglishApp(),
  ));
}

class EduHinglishApp extends ConsumerWidget {
  const EduHinglishApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return MaterialApp(
      title: 'EduHinglish',
      debugShowCheckedModeBanner: false,
      theme: buildTheme(Brightness.light),
      darkTheme: buildTheme(Brightness.dark),
      themeMode: ref.watch(settingsProvider.select((s) => s.themeMode)),
      // A shared-chat address opens the public viewer: no sign-in needed.
      home: switch (sharedTokenFromUri(Uri.base)) {
        final token? => SharedChatPage(token: token),
        _ => const _AuthGate(),
      },
    );
  }
}

/// Login page until signed in (skipped when the server has sign-in switched off).
class _AuthGate extends ConsumerWidget {
  const _AuthGate();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final status = ref.watch(authProvider.select((a) => a.status));
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 250),
      child: status == AuthStatus.signedIn ? const HomePage(key: ValueKey('home')) : const LoginPage(key: ValueKey('login')),
    );
  }
}
