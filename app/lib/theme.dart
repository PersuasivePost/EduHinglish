import 'package:flutter/cupertino.dart' show CupertinoPageTransitionsBuilder;
import 'package:flutter/material.dart';

/// Minimal palette: neutral greys plus one accent, light and dark. Follows the system setting by default.
const _accent = Color(0xFF3F51B5);

ThemeData buildTheme(Brightness b) {
  final dark = b == Brightness.dark;
  final base = ColorScheme.fromSeed(seedColor: _accent, brightness: b);
  final scheme = base.copyWith(
    primary: dark ? const Color(0xFFAEB8FF) : _accent,
    onPrimary: dark ? const Color(0xFF14172E) : Colors.white,
    surface: dark ? const Color(0xFF151517) : Colors.white,
    onSurface: dark ? const Color(0xFFECECF0) : const Color(0xFF1A1A1C),
    onSurfaceVariant: dark ? const Color(0xFF9A9AA4) : const Color(0xFF6B6B74),
    surfaceContainerLow: dark ? const Color(0xFF1B1B1E) : const Color(0xFFF7F7F8),
    surfaceContainer: dark ? const Color(0xFF202024) : const Color(0xFFF2F2F4),
    surfaceContainerHigh: dark ? const Color(0xFF28282D) : const Color(0xFFEDEDF0),
    surfaceContainerHighest: dark ? const Color(0xFF303036) : const Color(0xFFE6E6EA),
    secondaryContainer: dark ? const Color(0xFF2A2A30) : const Color(0xFFEAEAEE),
    onSecondaryContainer: dark ? const Color(0xFFECECF0) : const Color(0xFF1A1A1C),
    outlineVariant: dark ? const Color(0xFF2C2C32) : const Color(0xFFE4E4E8),
  );
  const smooth = PageTransitionsTheme(builders: {
    TargetPlatform.android: FadeForwardsPageTransitionsBuilder(),
    TargetPlatform.iOS: CupertinoPageTransitionsBuilder(),
  });
  return ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor: scheme.surface,
    pageTransitionsTheme: smooth,
    splashFactory: InkSparkle.splashFactory,
    appBarTheme: AppBarTheme(
      backgroundColor: scheme.surface,
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      scrolledUnderElevation: 0,
      titleTextStyle: TextStyle(color: scheme.onSurface, fontSize: 17, fontWeight: FontWeight.w600),
    ),
    dividerTheme: DividerThemeData(color: scheme.outlineVariant, space: 1),
    dialogTheme: DialogThemeData(backgroundColor: scheme.surface, surfaceTintColor: Colors.transparent),
    popupMenuTheme: PopupMenuThemeData(
        color: scheme.surfaceContainerLow, surfaceTintColor: Colors.transparent, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14))),
    snackBarTheme: SnackBarThemeData(
        behavior: SnackBarBehavior.floating, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))),
    inputDecorationTheme: const InputDecorationTheme(border: InputBorder.none),
  );
}
