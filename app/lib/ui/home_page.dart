import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../state.dart';
import 'chat_view.dart';
import 'share_sheet.dart';
import 'sidebar.dart';

const kWideBreakpoint = 900.0;

class HomePage extends ConsumerWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final wide = MediaQuery.sizeOf(context).width >= kWideBreakpoint;
    final chatId = ref.watch(sessionProvider.select((s) => s.chatId));
    final chats = ref.watch(chatsProvider).value ?? const [];
    final hasMessages = ref.watch(sessionProvider.select((s) => s.messages.any((m) => !m.streaming && !m.id.startsWith('local-'))));
    final title = chats.where((c) => c.id == chatId).map((c) => c.title).firstOrNull ?? 'EduHinglish';

    if (wide) {
      return Scaffold(
        body: Row(children: [
          const SizedBox(width: 300, child: Sidebar()),
          VerticalDivider(width: 1, color: Theme.of(context).colorScheme.outlineVariant),
          Expanded(
            child: Column(children: [
              SizedBox(
                height: 52,
                child: Row(children: [
                  const SizedBox(width: 20),
                  Expanded(child: Text(title, overflow: TextOverflow.ellipsis, style: Theme.of(context).appBarTheme.titleTextStyle)),
                  ..._actions(context, ref, chatId, hasMessages),
                  const SizedBox(width: 8),
                ]),
              ),
              const Expanded(child: ChatView()),
            ]),
          ),
        ]),
      );
    }
    return Scaffold(
      appBar: AppBar(
        title: Text(title, overflow: TextOverflow.ellipsis),
        actions: [
          ..._actions(context, ref, chatId, hasMessages),
          IconButton(
            tooltip: 'New chat',
            icon: const Icon(Icons.edit_square),
            onPressed: () => ref.read(sessionProvider.notifier).newChat(),
          ),
        ],
      ),
      drawer: const Drawer(child: Sidebar(closeOnSelect: true)),
      body: const ChatView(),
    );
  }
}

/// Copy and share buttons; only shown once the chat has answered messages.
List<Widget> _actions(BuildContext context, WidgetRef ref, String? chatId, bool hasMessages) {
  if (chatId == null || !hasMessages) return const [];
  return [
    IconButton(tooltip: 'Poori chat copy karo', icon: const Icon(Icons.copy_all_rounded), onPressed: () => copyChat(context, ref)),
    IconButton(tooltip: 'Share', icon: const Icon(Icons.ios_share_rounded), onPressed: () => showShareSheet(context)),
  ];
}
