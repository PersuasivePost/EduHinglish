import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';

const _suggestions = [
  ('Newton ka third law samjhao', Icons.rocket_launch_outlined),
  ('Photosynthesis kaise hota hai?', Icons.eco_outlined),
  ('Ohm ka law aur uska formula batao', Icons.bolt_outlined),
  ('Acids aur bases mein kya difference hai?', Icons.science_outlined),
];

class Welcome extends StatelessWidget {
  const Welcome({super.key, required this.onPick});

  final void Function(String question) onPick;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final text = Theme.of(context).textTheme;
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 640),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Icon(Icons.school_rounded, size: 56, color: cs.primary)
                .animate()
                .scale(begin: const Offset(0.6, 0.6), duration: 500.ms, curve: Curves.elasticOut)
                .fadeIn(),
            const SizedBox(height: 16),
            Text('Namaste! 👋', style: text.headlineMedium?.copyWith(fontWeight: FontWeight.w700))
                .animate()
                .fadeIn(delay: 150.ms)
                .slideY(begin: 0.2),
            const SizedBox(height: 6),
            Text('Class 9-10 NCERT ka koi bhi sawaal Hinglish mein pucho.',
                    textAlign: TextAlign.center, style: text.bodyLarge?.copyWith(color: cs.onSurfaceVariant))
                .animate()
                .fadeIn(delay: 250.ms),
            const SizedBox(height: 28),
            Wrap(spacing: 12, runSpacing: 12, alignment: WrapAlignment.center, children: [
              for (var i = 0; i < _suggestions.length; i++)
                _SuggestionCard(text: _suggestions[i].$1, icon: _suggestions[i].$2, onTap: () => onPick(_suggestions[i].$1))
                    .animate()
                    .fadeIn(delay: (350 + 90 * i).ms)
                    .slideY(begin: 0.25, curve: Curves.easeOutCubic),
            ]),
          ]),
        ),
      ),
    );
  }
}

class _SuggestionCard extends StatelessWidget {
  const _SuggestionCard({required this.text, required this.icon, required this.onTap});

  final String text;
  final IconData icon;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return SizedBox(
      width: 280,
      child: Material(
        color: cs.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(16),
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Row(children: [
              Icon(icon, color: cs.primary),
              const SizedBox(width: 12),
              Expanded(child: Text(text)),
            ]),
          ),
        ),
      ),
    );
  }
}
