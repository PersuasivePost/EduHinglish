import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../state.dart';
import 'composer.dart';
import 'message_bubble.dart';
import 'welcome.dart';

class ChatView extends ConsumerWidget {
  const ChatView({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final s = ref.watch(sessionProvider);
    final cs = Theme.of(context).colorScheme;
    final notifier = ref.read(sessionProvider.notifier);

    Widget body;
    if (s.loading) {
      body = const Center(child: CircularProgressIndicator());
    } else if (s.messages.isEmpty) {
      body = s.error != null
          ? _Banner(message: s.error!, onRetry: s.chatId != null ? () => notifier.open(s.chatId!) : null)
          : Welcome(onPick: notifier.send);
    } else {
      body = ListView.builder(
        reverse: true, // newest at the bottom; stays pinned there while the answer streams
        padding: const EdgeInsets.fromLTRB(12, 16, 12, 8),
        itemCount: s.messages.length,
        itemBuilder: (context, i) {
          final m = s.messages[s.messages.length - 1 - i];
          return Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 800),
              child: MessageBubble(key: ValueKey(m.id), message: m, stage: m.streaming ? s.stage : null),
            ),
          );
        },
      );
    }

    return Column(children: [
      Expanded(child: SelectionArea(child: body)), // drag to select and copy across messages
      if (s.error != null && s.messages.isNotEmpty)
        Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 800),
            child: _Banner(message: s.error!, onRetry: s.lastText != null ? notifier.retry : null, compact: true),
          ),
        ),
      Container(color: cs.surface, child: const Composer()),
    ]);
  }
}

class _Banner extends StatelessWidget {
  const _Banner({required this.message, this.onRetry, this.compact = false});

  final String message;
  final VoidCallback? onRetry;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final card = Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(color: cs.errorContainer, borderRadius: BorderRadius.circular(14)),
      child: Row(children: [
        Icon(Icons.error_outline, color: cs.onErrorContainer),
        const SizedBox(width: 10),
        Expanded(child: Text(message, style: TextStyle(color: cs.onErrorContainer))),
        if (onRetry != null) TextButton(onPressed: onRetry, child: const Text('Retry')),
      ]),
    );
    return compact ? card : Center(child: ConstrainedBox(constraints: const BoxConstraints(maxWidth: 560), child: card));
  }
}
