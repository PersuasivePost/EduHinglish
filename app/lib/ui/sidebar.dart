import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models.dart';
import '../auth.dart';
import '../state.dart';
import 'settings_dialog.dart';

class Sidebar extends ConsumerWidget {
  const Sidebar({super.key, this.closeOnSelect = false});

  final bool closeOnSelect;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final chats = ref.watch(chatsProvider);
    final selected = ref.watch(sessionProvider.select((s) => s.chatId));
    final cs = Theme.of(context).colorScheme;

    void close() {
      if (closeOnSelect) Navigator.of(context).pop();
    }

    return Material(
      color: cs.surfaceContainerLow,
      child: SafeArea(
        child: Column(children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: Row(children: [
              Icon(Icons.school_rounded, color: cs.primary, size: 28),
              const SizedBox(width: 10),
              Flexible(
                child: Text('EduHinglish',
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
              ),
            ]),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: SizedBox(
              width: double.infinity,
              child: FilledButton.tonalIcon(
                icon: const Icon(Icons.add),
                label: const Text('Naya chat'),
                onPressed: () {
                  ref.read(sessionProvider.notifier).newChat();
                  close();
                },
              ),
            ),
          ),
          Expanded(
            child: chats.when(
              loading: () => const Center(child: CircularProgressIndicator()),
              error: (e, _) => _ListError(message: e.toString(), onRetry: () => ref.invalidate(chatsProvider)),
              data: (list) => list.isEmpty
                  ? Center(
                      child: Padding(
                        padding: const EdgeInsets.all(24),
                        child: Text('Abhi koi chat nahi.\nEk sawaal pooch ke shuru karo!',
                            textAlign: TextAlign.center, style: TextStyle(color: cs.onSurfaceVariant)),
                      ),
                    )
                  : RefreshIndicator(
                      onRefresh: () => ref.read(chatsProvider.notifier).refresh(),
                      child: ListView.builder(
                        padding: const EdgeInsets.symmetric(horizontal: 8),
                        itemCount: list.length,
                        itemBuilder: (context, i) => _ChatTile(
                          chat: list[i],
                          selected: list[i].id == selected,
                          onTap: () {
                            ref.read(sessionProvider.notifier).open(list[i].id);
                            close();
                          },
                        ).animate().fadeIn(duration: 220.ms, delay: (30 * i.clamp(0, 10)).ms).slideX(begin: -0.05),
                      ),
                    ),
            ),
          ),
          const Divider(height: 1),
          const _UserTile(),
        ]),
      ),
    );
  }
}

class _ChatTile extends ConsumerWidget {
  const _ChatTile({required this.chat, required this.selected, required this.onTap});

  final Chat chat;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final cs = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 1),
      child: ListTile(
        dense: true,
        selected: selected,
        selectedTileColor: cs.secondaryContainer,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        title: Text(chat.title, maxLines: 1, overflow: TextOverflow.ellipsis),
        onTap: onTap,
        trailing: PopupMenuButton<String>(
          icon: const Icon(Icons.more_horiz, size: 20),
          tooltip: 'Options',
          onSelected: (v) async {
            if (v == 'rename') {
              final title = await _askTitle(context, chat.title);
              if (title != null && context.mounted) await _guard(context, () => ref.read(chatsProvider.notifier).rename(chat.id, title));
            } else if (v == 'delete') {
              await _guard(context, () => ref.read(chatsProvider.notifier).delete(chat.id));
            }
          },
          itemBuilder: (_) => const [
            PopupMenuItem(value: 'rename', child: Text('Rename')),
            PopupMenuItem(value: 'delete', child: Text('Delete')),
          ],
        ),
      ),
    );
  }

  Future<void> _guard(BuildContext context, Future<void> Function() action) async {
    try {
      await action();
    } catch (e) {
      if (context.mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.toString())));
    }
  }

  Future<String?> _askTitle(BuildContext context, String current) {
    final c = TextEditingController(text: current);
    return showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Rename chat'),
        content: TextField(controller: c, autofocus: true, maxLength: 120, onSubmitted: (v) => Navigator.pop(ctx, v.trim())),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, c.text.trim()), child: const Text('Save')),
        ],
      ),
    ).then((v) => (v == null || v.isEmpty) ? null : v);
  }
}

class _ListError extends StatelessWidget {
  const _ListError({required this.message, required this.onRetry});
  final String message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) => Center(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Icon(Icons.cloud_off, color: Theme.of(context).colorScheme.error),
            const SizedBox(height: 8),
            Text(message, textAlign: TextAlign.center),
            const SizedBox(height: 8),
            OutlinedButton(onPressed: onRetry, child: const Text('Retry')),
          ]),
        ),
      );
}

class _UserTile extends ConsumerWidget {
  const _UserTile();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final user = ref.watch(authProvider.select((a) => a.user));
    final signedInWithGoogle = ref.watch(authProvider.select((a) => a.token != null));
    final cs = Theme.of(context).colorScheme;
    final name = user?.name ?? user?.email ?? 'Guest';
    return PopupMenuButton<String>(
      tooltip: 'Account',
      position: PopupMenuPosition.over,
      onSelected: (v) {
        if (v == 'settings') showDialog(context: context, builder: (_) => const SettingsDialog());
        if (v == 'signout') ref.read(authProvider.notifier).signOut();
      },
      itemBuilder: (_) => [
        const PopupMenuItem(value: 'settings', child: ListTile(leading: Icon(Icons.settings_outlined), title: Text('Settings'), dense: true)),
        if (signedInWithGoogle)
          const PopupMenuItem(value: 'signout', child: ListTile(leading: Icon(Icons.logout_rounded), title: Text('Sign out'), dense: true)),
      ],
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(children: [
          CircleAvatar(
            radius: 16,
            backgroundColor: cs.surfaceContainerHighest,
            foregroundImage: (user?.picture ?? '').isEmpty ? null : NetworkImage(user!.picture!),
            child: Text(name.characters.first.toUpperCase(), style: TextStyle(color: cs.onSurface)),
          ),
          const SizedBox(width: 12),
          Expanded(child: Text(name, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontWeight: FontWeight.w600))),
          Icon(Icons.unfold_more_rounded, size: 18, color: cs.onSurfaceVariant),
        ]),
      ),
    );
  }
}
