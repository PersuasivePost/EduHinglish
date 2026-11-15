import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';

const _stageText = {
  'queued': 'Line mein hain, thoda ruko…',
  'rewrite': 'Sawaal samajh raha hoon…',
  'retrieve': 'NCERT mein dhoondh raha hoon…',
  'generate': 'Jawab likh raha hoon…',
};

/// Three bouncing dots plus what the system is doing right now.
class TypingIndicator extends StatelessWidget {
  const TypingIndicator({super.key, this.stage});

  final String? stage;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Row(mainAxisSize: MainAxisSize.min, children: [
      for (var i = 0; i < 3; i++)
        Container(
          width: 7,
          height: 7,
          margin: const EdgeInsets.only(right: 4),
          decoration: BoxDecoration(color: cs.primary, shape: BoxShape.circle),
        )
            .animate(onPlay: (c) => c.repeat(reverse: true), delay: (i * 160).ms)
            .moveY(begin: 2, end: -3, duration: 380.ms, curve: Curves.easeInOut)
            .fade(begin: 0.4, end: 1, duration: 380.ms),
      const SizedBox(width: 6),
      AnimatedSwitcher(
        duration: const Duration(milliseconds: 250),
        child: Text(
          _stageText[stage] ?? 'Soch raha hoon…',
          key: ValueKey(stage),
          style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13),
        ),
      ),
    ]);
  }
}
