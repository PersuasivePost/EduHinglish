import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models.dart';
import '../state.dart';
import 'typing_indicator.dart';

class MessageBubble extends ConsumerWidget {
  const MessageBubble({super.key, required this.message, this.stage});

  final Message message;
  final String? stage;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final bubble = message.isUser ? _UserBubble(message: message) : _AssistantBubble(message: message, stage: stage);
    // Slide + fade in once, when the bubble first appears.
    return bubble.animate().fadeIn(duration: 260.ms).slideY(begin: 0.08, duration: 260.ms, curve: Curves.easeOutCubic);
  }
}

class _UserBubble extends StatelessWidget {
  const _UserBubble({required this.message});
  final Message message;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Align(
      alignment: Alignment.centerRight,
      child: Container(
        margin: const EdgeInsets.only(left: 56, top: 6, bottom: 6),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: cs.surfaceContainerHigh,
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(20),
            topRight: Radius.circular(20),
            bottomLeft: Radius.circular(20),
            bottomRight: Radius.circular(4),
          ),
          border: message.failed ? Border.all(color: cs.error) : null,
        ),
        child: Text(message.content, style: TextStyle(color: cs.onSurface, fontSize: 15.5, height: 1.4)),
      ),
    );
  }
}

class _AssistantBubble extends ConsumerStatefulWidget {
  const _AssistantBubble({required this.message, this.stage});
  final Message message;
  final String? stage;

  @override
  ConsumerState<_AssistantBubble> createState() => _AssistantBubbleState();
}

class _AssistantBubbleState extends ConsumerState<_AssistantBubble> {
  bool _showSources = false;

  @override
  Widget build(BuildContext context) {
    final m = widget.message;
    final cs = Theme.of(context).colorScheme;
    final waiting = m.streaming && m.content.isEmpty;
    final settled = !m.streaming && !m.id.startsWith('local-');

    return Align(
      alignment: Alignment.centerLeft,
      child: Padding(
        padding: const EdgeInsets.only(right: 24, top: 6, bottom: 6),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          CircleAvatar(radius: 16, backgroundColor: cs.primary, child: Icon(Icons.school_rounded, size: 18, color: cs.onPrimary)),
          const SizedBox(width: 10),
          Flexible(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              if (waiting)
                Padding(padding: const EdgeInsets.only(top: 8), child: TypingIndicator(stage: widget.stage))
              else
                MarkdownBody(
                  data: m.streaming ? '${m.content} ▍' : m.content,
                  selectable: false, // the whole chat sits in one SelectionArea (see chat_view.dart)
                  styleSheet: MarkdownStyleSheet.fromTheme(Theme.of(context)).copyWith(
                    p: TextStyle(fontSize: 15.5, height: 1.5, color: cs.onSurface),
                  ),
                ),
              if (settled) _actions(m, cs),
              if (_showSources && m.sources.isNotEmpty) _sources(m, cs),
            ]),
          ),
        ]),
      ),
    );
  }

  Widget _actions(Message m, ColorScheme cs) {
    final notifier = ref.read(sessionProvider.notifier);
    Widget btn(IconData icon, String tip, VoidCallback onTap, {bool active = false}) => IconButton(
          visualDensity: VisualDensity.compact,
          iconSize: 18,
          tooltip: tip,
          icon: Icon(icon, color: active ? cs.primary : cs.onSurfaceVariant),
          onPressed: onTap,
        );
    return Padding(
      padding: const EdgeInsets.only(top: 4),
      child: Wrap(crossAxisAlignment: WrapCrossAlignment.center, children: [
        btn(Icons.copy_rounded, 'Copy', () {
          Clipboard.setData(ClipboardData(text: m.content));
          ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Copy ho gaya'), duration: Duration(seconds: 1)));
        }),
        btn(m.feedback == 1 ? Icons.thumb_up_alt : Icons.thumb_up_alt_outlined, 'Helpful', () => notifier.setFeedback(m, 1),
            active: m.feedback == 1),
        btn(m.feedback == -1 ? Icons.thumb_down_alt : Icons.thumb_down_alt_outlined, 'Not helpful',
            () => notifier.setFeedback(m, -1), active: m.feedback == -1),
        // Placeholder for the last project stage (text to speech).
        btn(Icons.volume_up_outlined, 'Listen (coming soon)', () {}),
        if (m.sources.isNotEmpty)
          TextButton.icon(
            onPressed: () => setState(() => _showSources = !_showSources),
            icon: Icon(_showSources ? Icons.expand_less : Icons.menu_book_outlined, size: 16),
            label: Text('NCERT sources (${m.sources.length})'),
            style: TextButton.styleFrom(visualDensity: VisualDensity.compact, foregroundColor: cs.onSurfaceVariant),
          ),
      ]),
    );
  }

  Widget _sources(Message m, ColorScheme cs) => Padding(
        padding: const EdgeInsets.only(top: 4),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          for (final s in m.sources)
            Container(
              margin: const EdgeInsets.only(bottom: 6),
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(color: cs.surfaceContainerHigh, borderRadius: BorderRadius.circular(12)),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('${s.chapter}${s.classNum.isEmpty ? '' : '  •  Class ${s.classNum}'}',
                    style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13)),
                if (s.textPreview.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Text(s.textPreview, style: TextStyle(fontSize: 12.5, color: cs.onSurfaceVariant)),
                  ),
              ]),
            ).animate().fadeIn(duration: 200.ms),
        ]),
      );
}
