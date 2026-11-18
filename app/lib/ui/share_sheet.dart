import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models.dart';
import '../state.dart';

/// The whole chat as plain text, for pasting anywhere.
String chatAsText(List<Message> messages) => messages
    .where((m) => !m.streaming && m.content.isNotEmpty)
    .map((m) => '${m.isUser ? 'Student' : 'EduHinglish'}: ${m.content}')
    .join('\n\n');

void _toast(BuildContext context, String text) =>
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text), duration: const Duration(seconds: 2)));

Future<void> copyChat(BuildContext context, WidgetRef ref) async {
  final text = chatAsText(ref.read(sessionProvider).messages);
  if (text.isEmpty) return;
  await Clipboard.setData(ClipboardData(text: text));
  if (context.mounted) _toast(context, 'Poori chat copy ho gayi');
}

Future<void> showShareSheet(BuildContext context) => showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      constraints: const BoxConstraints(maxWidth: 560),
      builder: (_) => const _ShareSheet(),
    );

class _ShareSheet extends ConsumerStatefulWidget {
  const _ShareSheet();

  @override
  ConsumerState<_ShareSheet> createState() => _ShareSheetState();
}

class _ShareSheetState extends ConsumerState<_ShareSheet> {
  bool _busy = false;
  String? _error;

  Future<void> _run(Future<void> Function() action) async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await action();
    } catch (e) {
      _error = e.toString();
    }
    if (mounted) setState(() => _busy = false);
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final chatId = ref.watch(sessionProvider.select((s) => s.chatId));
    final chats = ref.watch(chatsProvider).value ?? const <Chat>[];
    final chat = chats.where((c) => c.id == chatId).firstOrNull;
    final token = chat?.shareToken;
    final api = ref.read(apiProvider);
    final link = token == null ? null : api.shareUrl(token);

    return Padding(
      padding: EdgeInsets.fromLTRB(20, 0, 20, 20 + MediaQuery.viewInsetsOf(context).bottom),
      child: AnimatedSize(
        duration: const Duration(milliseconds: 220),
        curve: Curves.easeOutCubic,
        child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('Chat share karo', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: 6),
          Text(
            link == null
                ? 'Link banane par is chat ka abhi tak ka hissa public ho jaata hai. Link wala koi bhi ise padh sakta hai '
                    '(sirf padh sakta hai, sign in nahi chahiye). Baad ke messages tab dikhenge jab tum link refresh karo.'
                : 'Link wala koi bhi is chat ko padh sakta hai. Naye messages dikhane ke liye "Refresh link" dabao.',
            style: TextStyle(color: cs.onSurfaceVariant, height: 1.4),
          ),
          const SizedBox(height: 16),
          if (link != null) ...[
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              decoration: BoxDecoration(color: cs.surfaceContainerHigh, borderRadius: BorderRadius.circular(12)),
              child: SelectableText(link, style: const TextStyle(fontSize: 14)),
            ),
            const SizedBox(height: 12),
            Wrap(spacing: 8, runSpacing: 8, children: [
              FilledButton.icon(
                icon: const Icon(Icons.link_rounded),
                label: const Text('Link copy karo'),
                onPressed: () async {
                  await Clipboard.setData(ClipboardData(text: link));
                  if (context.mounted) _toast(context, 'Link copy ho gaya');
                },
              ),
              OutlinedButton.icon(
                icon: const Icon(Icons.refresh_rounded),
                label: const Text('Refresh link'),
                onPressed: _busy ? null : () => _run(() async => await ref.read(chatsProvider.notifier).share(chatId!)),
              ),
              TextButton(
                onPressed: _busy ? null : () => _run(() => ref.read(chatsProvider.notifier).unshare(chatId!)),
                child: const Text('Sharing band karo'),
              ),
            ]),
          ] else
            FilledButton.icon(
              icon: _busy
                  ? const SizedBox(height: 16, width: 16, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.ios_share_rounded),
              label: const Text('Link banao'),
              onPressed: _busy || chatId == null ? null : () => _run(() async => await ref.read(chatsProvider.notifier).share(chatId)),
            ),
          if (_error != null) Padding(padding: const EdgeInsets.only(top: 10), child: Text(_error!, style: TextStyle(color: cs.error))),
        ]),
      ),
    );
  }
}
