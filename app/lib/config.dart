/// Backend address. Override at build time:
///   flutter run -d chrome --dart-define=BACKEND_URL=https://eduhinglish.onrender.com
/// It can also be changed inside the app (Settings → Server URL), which is handy for phones.
const String _definedUrl = String.fromEnvironment('BACKEND_URL');

/// --dart-define wins; otherwise the deployed backend. For a backend on your own machine use
/// --dart-define=BACKEND_URL=http://10.0.2.2:8000 (Android emulator) or http://localhost:8000 (web/desktop).
String get kDefaultBackendUrl => _definedUrl.isNotEmpty ? _definedUrl : _renderUrl;

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
