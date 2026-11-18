import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:url_launcher/url_launcher.dart';

import '../api.dart';
import '../models.dart';
import '../state.dart';
import 'share_sheet.dart' show chatAsText;

final _sharedProvider = FutureProvider.family<(Chat, List<Message>), String>(
    (ref, token) => ApiClient(ref.read(settingsProvider).backendUrl).sharedChat(token));

/// Read-only view of a shared chat at `/s/<token>`. No sign-in, no model server: it only reads the database.
class SharedChatPage extends ConsumerWidget {
  const SharedChatPage({super.key, required this.token});

  final String token;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final cs = Theme.of(context).colorScheme;
    final data = ref.watch(_sharedProvider(token));
    final msgs = data.value?.$2 ?? const <Message>[];

    return Scaffold(
      appBar: AppBar(
        title: Text(data.value?.$1.title ?? 'EduHinglish', overflow: TextOverflow.ellipsis),
        actions: [
          if (msgs.isNotEmpty)
            IconButton(
              tooltip: 'Poori chat copy karo',
              icon: const Icon(Icons.copy_all_rounded),
              onPressed: () async {
                await Clipboard.setData(ClipboardData(text: chatAsText(msgs)));
                if (context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Poori chat copy ho gayi')));
                }
              },
            ),
        ],
      ),
      body: data.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => _Unavailable(
          message: e is ApiException && e.status == 404
              ? 'Yeh link ab available nahi hai. Ho sakta hai chat ka sharing band kar diya gaya ho.'
              : e.toString(),
          onRetry: e is ApiException && e.status == 404 ? null : () => ref.invalidate(_sharedProvider(token)),
        ),
        data: (d) => SelectionArea(
          child: ListView(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
            children: [
              for (final (i, m) in d.$2.indexed)
                Center(
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 800),
                    child: _SharedBubble(message: m)
                        .animate()
                        .fadeIn(delay: (40 * i.clamp(0, 12)).ms, duration: 240.ms)
                        .slideY(begin: 0.05, curve: Curves.easeOutCubic),
                  ),
                ),
              const SizedBox(height: 24),
              Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 800),
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(color: cs.surfaceContainerLow, borderRadius: BorderRadius.circular(16)),
                    child: Row(children: [
                      Icon(Icons.school_rounded, color: cs.primary),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text('Shared EduHinglish chat · sirf padh sakte ho. Jawab mein galtiyan ho sakti hain, '
                            'apni NCERT kitab se check karo.',
                            style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13, height: 1.4)),
                      ),
                      const SizedBox(width: 12),
                      FilledButton(
                        onPressed: () => launchUrl(Uri.base.replace(path: '/', query: '', fragment: ''),
                            webOnlyWindowName: '_self'),
                        child: const Text('EduHinglish kholo'),
                      ),
                    ]),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _SharedBubble extends StatelessWidget {
  const _SharedBubble({required this.message});
  final Message message;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    if (message.isUser) {
      return Align(
        alignment: Alignment.centerRight,
        child: Container(
          margin: const EdgeInsets.only(left: 56, top: 6, bottom: 6),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          decoration: BoxDecoration(color: cs.surfaceContainerHigh, borderRadius: BorderRadius.circular(20)),
          child: Text(message.content, style: TextStyle(color: cs.onSurface, fontSize: 15.5, height: 1.4)),
        ),
      );
    }
    final chapters = {for (final s in message.sources) s.chapter}.where((c) => c.isNotEmpty).toList();
    return Padding(
      padding: const EdgeInsets.only(right: 24, top: 6, bottom: 6),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        CircleAvatar(radius: 16, backgroundColor: cs.primary, child: Icon(Icons.school_rounded, size: 18, color: cs.onPrimary)),
        const SizedBox(width: 10),
        Flexible(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            MarkdownBody(
              data: message.content,
              styleSheet: MarkdownStyleSheet.fromTheme(Theme.of(context))
                  .copyWith(p: TextStyle(fontSize: 15.5, height: 1.5, color: cs.onSurface)),
            ),
            if (chapters.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text('NCERT: ${chapters.join(', ')}', style: TextStyle(fontSize: 12.5, color: cs.onSurfaceVariant)),
              ),
          ]),
        ),
      ]),
    );
  }
}

class _Unavailable extends StatelessWidget {
  const _Unavailable({required this.message, this.onRetry});
  final String message;
  final VoidCallback? onRetry;

  @override
  Widget build(BuildContext context) => Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Icon(Icons.link_off_rounded, size: 40, color: Theme.of(context).colorScheme.onSurfaceVariant),
            const SizedBox(height: 12),
            Text(message, textAlign: TextAlign.center),
            if (onRetry != null) Padding(padding: const EdgeInsets.only(top: 12), child: OutlinedButton(onPressed: onRetry, child: const Text('Retry'))),
          ]),
        ),
      );
}
