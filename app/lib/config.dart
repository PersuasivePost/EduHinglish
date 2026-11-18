import 'package:flutter/foundation.dart' show kIsWeb;

/// Backend address. Override at build time:
///   flutter run -d chrome --dart-define=BACKEND_URL=https://eduhinglish.onrender.com
/// It can also be changed inside the app (Settings → Server URL), which is handy for phones.
const String _definedUrl = String.fromEnvironment('BACKEND_URL');

/// --dart-define wins; otherwise the deployed backend. For a backend on your own machine use
/// --dart-define=BACKEND_URL=http://10.0.2.2:8000 (Android emulator) or http://localhost:8000 (web/desktop).
String get kDefaultBackendUrl => _definedUrl.isNotEmpty ? _definedUrl : _renderUrl;

/// Where shared-chat links point. On the web build this is the site itself (`Uri.base.origin`); for an APK pass
///   --dart-define=WEB_URL=https://your-site.vercel.app
/// Without either, links use the backend's own plain page (`<backend>/s/<token>`).
const String _definedWebUrl = String.fromEnvironment('WEB_URL');

String shareLink(String token, String backendUrl) {
  final site = _definedWebUrl.isNotEmpty ? _definedWebUrl : (kIsWeb ? Uri.base.origin : backendUrl);
  return '${site.replaceAll(RegExp(r'/+$'), '')}/s/$token';
}

/// The token in a shared-chat address (`/s/<token>`), or null for any other address.
String? sharedTokenFromUri(Uri uri) {
  final seg = uri.pathSegments;
  return seg.length == 2 && seg[0] == 's' && seg[1].isNotEmpty ? seg[1] : null;
}

const String _renderUrl = 'https://eduhinglish.onrender.com';

const subjects = <String, String>{
  'biology': 'Biology',
  'chemistry': 'Chemistry',
  'physics': 'Physics',
  'mathematics': 'Maths',
  'social_science': 'Social Science',
};

const classes = <String>['9', '10'];

const languages = <String, String>{
  'hinglish': 'Hinglish',
  'english': 'English',
};
