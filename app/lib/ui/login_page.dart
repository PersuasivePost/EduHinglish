import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../auth.dart';
import 'google_button_stub.dart' if (dart.library.js_interop) 'google_button_web.dart';
import 'settings_dialog.dart';

class LoginPage extends ConsumerWidget {
  const LoginPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final auth = ref.watch(authProvider);
    final cs = Theme.of(context).colorScheme;
    final text = Theme.of(context).textTheme;
    final webButton = buildWebGoogleButton();

    Widget action;
    if (auth.status == AuthStatus.loading) {
      action = const SizedBox(height: 28, width: 28, child: CircularProgressIndicator(strokeWidth: 2.5));
    } else if (auth.status == AuthStatus.error) {
      action = Column(children: [
        Text(auth.error ?? 'Kuch gadbad ho gayi', textAlign: TextAlign.center, style: TextStyle(color: cs.error)),
        const SizedBox(height: 12),
        OutlinedButton(onPressed: ref.read(authProvider.notifier).retry, child: const Text('Retry')),
      ]);
    } else {
      action = Column(children: [
        if (webButton != null)
          SizedBox(height: 44, child: webButton)
        else
          FilledButton.icon(
            onPressed: auth.busy ? null : ref.read(authProvider.notifier).signInWithGoogle,
            icon: const Icon(Icons.login_rounded),
            label: const Text('Google se sign in karo'),
            style: FilledButton.styleFrom(minimumSize: const Size(240, 48)),
          ),
        if ((auth.error ?? '').isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 14),
            child: Text(auth.error!, textAlign: TextAlign.center, style: TextStyle(color: cs.error)),
          ),
      ]);
    }

    return Scaffold(
      body: SafeArea(
        child: Stack(children: [
          Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(28),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 380),
                child: Column(mainAxisSize: MainAxisSize.min, children: [
                  Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(color: cs.primary, shape: BoxShape.circle),
                    child: Icon(Icons.school_rounded, size: 40, color: cs.onPrimary),
                  ).animate().scale(begin: const Offset(0.7, 0.7), duration: 450.ms, curve: Curves.easeOutBack).fadeIn(),
                  const SizedBox(height: 24),
                  Text('EduHinglish', style: text.headlineMedium?.copyWith(fontWeight: FontWeight.w700))
                      .animate()
                      .fadeIn(delay: 120.ms)
                      .slideY(begin: 0.2),
                  const SizedBox(height: 8),
                  Text('NCERT Class 9-10 ka AI tutor, Hinglish mein.',
                          textAlign: TextAlign.center, style: text.bodyLarge?.copyWith(color: cs.onSurfaceVariant))
                      .animate()
                      .fadeIn(delay: 220.ms),
                  const SizedBox(height: 36),
                  action.animate().fadeIn(delay: 320.ms).slideY(begin: 0.2),
                  const SizedBox(height: 16),
                  Text('Sign in karne se tumhari chats sirf tumhe dikhengi, kisi bhi device par.',
                          textAlign: TextAlign.center, style: text.bodySmall?.copyWith(color: cs.onSurfaceVariant))
                      .animate()
                      .fadeIn(delay: 420.ms),
                ]),
              ),
            ),
          ),
          Positioned(
            top: 4,
            right: 4,
            child: IconButton(
              tooltip: 'Server settings',
              icon: const Icon(Icons.settings_outlined),
              onPressed: () => showDialog(context: context, builder: (_) => const SettingsDialog()),
            ),
          ),
        ]),
      ),
    );
  }
}
