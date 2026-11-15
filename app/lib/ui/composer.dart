import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../config.dart';
import '../state.dart';

class Composer extends ConsumerStatefulWidget {
  const Composer({super.key});

  @override
  ConsumerState<Composer> createState() => _ComposerState();
}

class _ComposerState extends ConsumerState<Composer> {
  final _controller = TextEditingController();
  final _focus = FocusNode();
  bool _hasText = false;

  @override
  void initState() {
    super.initState();
    _controller.addListener(() {
      final has = isAskable(_controller.text);
      if (has != _hasText) setState(() => _hasText = has);
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _focus.dispose();
    super.dispose();
  }

  void _send() {
    final text = _controller.text.trim();
    if (!isAskable(text)) return;
    ref.read(sessionProvider.notifier).send(text);
    _controller.clear();
    _focus.requestFocus();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final sending = ref.watch(sessionProvider.select((s) => s.sending));
    final settings = ref.watch(settingsProvider);
    final notifier = ref.read(settingsProvider.notifier);

    return SafeArea(
      top: false,
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 800),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 4, 12, 10),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              SizedBox(
                height: 40,
                child: ListView(scrollDirection: Axis.horizontal, children: [
                  _Pick<String?>(
                    icon: Icons.menu_book_outlined,
                    label: subjects[settings.subject] ?? 'All subjects',
                    options: {null: 'All subjects', ...subjects},
                    onSelected: notifier.setSubject,
                  ),
                  _Pick<String?>(
                    icon: Icons.school_outlined,
                    label: settings.classNum == null ? 'Class 9 & 10' : 'Class ${settings.classNum}',
                    options: {null: 'Class 9 & 10', for (final c in classes) c: 'Class $c'},
                    onSelected: notifier.setClass,
                  ),
                  _Pick<String>(
                    icon: Icons.translate,
                    label: languages[settings.language] ?? 'Hinglish',
                    options: languages,
                    onSelected: (v) => notifier.setLanguage(v ?? 'hinglish'),
                  ),
                ]),
              ),
              Container(
                decoration: BoxDecoration(
                  color: cs.surfaceContainerHigh,
                  borderRadius: BorderRadius.circular(28),
                ),
                padding: const EdgeInsets.fromLTRB(8, 4, 8, 4),
                child: Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
                  // Placeholder for the last project stage (speech to text).
                  IconButton(tooltip: 'Voice input (coming soon)', icon: const Icon(Icons.mic_none), onPressed: null),
                  Expanded(
                    child: Focus(
                      // Enter sends, Shift+Enter adds a new line (desktop / web keyboards)
                      onKeyEvent: (node, event) {
                        if (event is KeyDownEvent &&
                            event.logicalKey == LogicalKeyboardKey.enter &&
                            !HardwareKeyboard.instance.isShiftPressed &&
                            !sending) {
                          _send();
                          return KeyEventResult.handled;
                        }
                        return KeyEventResult.ignored;
                      },
                      child: TextField(
                        controller: _controller,
                        focusNode: _focus,
                        minLines: 1,
                        maxLines: 5,
                        maxLength: 1000,
                        buildCounter: (_, {required currentLength, required isFocused, maxLength}) => null,
                        keyboardType: TextInputType.multiline,
                        textCapitalization: TextCapitalization.sentences,
                        decoration: const InputDecoration(hintText: 'Apna sawaal yahan likho…', contentPadding: EdgeInsets.symmetric(vertical: 12)),
                      ),
                    ),
                  ),
                  AnimatedSwitcher(
                    duration: const Duration(milliseconds: 200),
                    transitionBuilder: (c, a) => ScaleTransition(scale: a, child: c),
                    child: sending
                        ? IconButton.filled(
                            key: const ValueKey('stop'),
                            tooltip: 'Stop',
                            icon: const Icon(Icons.stop_rounded),
                            onPressed: ref.read(sessionProvider.notifier).stop,
                          )
                        : IconButton.filled(
                            key: const ValueKey('send'),
                            tooltip: 'Send',
                            icon: const Icon(Icons.arrow_upward_rounded),
                            onPressed: _hasText ? _send : null,
                          ),
                  ),
                ]),
              ),
            ]),
          ),
        ),
      ),
    );
  }
}

/// Small chip that opens a menu of options.
class _Pick<T> extends StatelessWidget {
  const _Pick({required this.icon, required this.label, required this.options, required this.onSelected});

  final IconData icon;
  final String label;
  final Map<T, String> options;
  final void Function(T?) onSelected;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: PopupMenuButton<T>(
        tooltip: label,
        onSelected: onSelected,
        itemBuilder: (_) => [for (final e in options.entries) PopupMenuItem<T>(value: e.key, child: Text(e.value))],
        child: Chip(avatar: Icon(icon, size: 16), label: Text(label), visualDensity: VisualDensity.compact),
      ),
    );
  }
}
