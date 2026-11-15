import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_sign_in/google_sign_in.dart';

import 'api.dart';
import 'models.dart';
import 'state.dart';

enum AuthStatus { loading, signedOut, signedIn, error }

class AuthState {
  const AuthState({
    this.status = AuthStatus.loading,
    this.user,
    this.token,
    this.googleClientId,
    this.busy = false,
    this.error,
  });

  final AuthStatus status;
  final AppUser? user;
  final String? token;

  /// Web OAuth client id from the backend; null while the server has sign-in switched off.
  final String? googleClientId;
  final bool busy;
  final String? error;

  AuthState copyWith({
    AuthStatus? status,
    AppUser? user,
    String? token,
    String? googleClientId,
    bool? busy,
    String? error,
    bool clearError = false,
    bool clearSession = false,
  }) =>
      AuthState(
        status: status ?? this.status,
        user: clearSession ? null : (user ?? this.user),
        token: clearSession ? null : (token ?? this.token),
        googleClientId: googleClientId ?? this.googleClientId,
        busy: busy ?? this.busy,
        error: clearError ? null : (error ?? this.error),
      );
}

const _tokenKey = 'sessionToken';
bool _googleReady = false;

/// Sign-in state. When the backend has no GOOGLE_CLIENT_IDS (local testing) the app skips the login page.
class AuthNotifier extends Notifier<AuthState> {
  StreamSubscription<GoogleSignInAuthenticationEvent>? _events;

  @override
  AuthState build() {
    final url = ref.watch(settingsProvider.select((s) => s.backendUrl)); // new server -> start over
    ref.onDispose(() => _events?.cancel());
    Future.microtask(() => _start(url));
    return const AuthState();
  }

  ApiClient _api(String url, [String? token]) => ApiClient(url, token: token);

  Future<void> _start(String url) async {
    final prefs = ref.read(sharedPrefsProvider);
    final api = _api(url);
    try {
      final cfg = await api.authConfig();
      if (!cfg.enabled) {
        state = const AuthState(status: AuthStatus.signedIn, user: AppUser(id: 'local', name: 'Local user'));
        return;
      }
      await _initGoogle(cfg.googleClientId!);
      final token = prefs.getString(_tokenKey);
      if (token == null) {
        state = AuthState(status: AuthStatus.signedOut, googleClientId: cfg.googleClientId);
        return;
      }
      try {
        final user = await _api(url, token).me();
        state = AuthState(status: AuthStatus.signedIn, user: user, token: token, googleClientId: cfg.googleClientId);
      } on ApiException catch (e) {
        if (e.status == 401) {
          await prefs.remove(_tokenKey);
          state = AuthState(status: AuthStatus.signedOut, googleClientId: cfg.googleClientId);
        } else {
          rethrow;
        }
      }
    } on ApiException catch (e) {
      state = AuthState(status: AuthStatus.error, error: e.message);
    } catch (e) {
      state = AuthState(status: AuthStatus.error, error: 'Sign-in set up nahi ho paaya: $e');
    }
  }

  Future<void> _initGoogle(String clientId) async {
    // On Android/iOS the id token's audience is the *server* client id, on web it is the web client id.
    if (!_googleReady) {
      await GoogleSignIn.instance.initialize(
        clientId: kIsWeb ? clientId : null,
        serverClientId: kIsWeb ? null : clientId,
      );
      _googleReady = true;
    }
    await _events?.cancel();
    _events = GoogleSignIn.instance.authenticationEvents.listen((e) {
      if (e is GoogleSignInAuthenticationEventSignIn) _exchange(e.user.authentication.idToken);
    }, onError: (Object e) => state = state.copyWith(busy: false, error: _friendly(e)));
  }

  /// Mobile: opens the Google account chooser. (On web the rendered Google button does this instead.)
  Future<void> signInWithGoogle() async {
    state = state.copyWith(busy: true, clearError: true);
    try {
      final account = await GoogleSignIn.instance.authenticate();
      await _exchange(account.authentication.idToken);
    } catch (e) {
      state = state.copyWith(busy: false, error: _friendly(e));
    }
  }

  Future<void> _exchange(String? idToken) async {
    if (idToken == null) {
      state = state.copyWith(busy: false, error: 'Google ne token nahi diya. Dobara try karo.');
      return;
    }
    state = state.copyWith(busy: true, clearError: true);
    try {
      final (token, user) = await _api(ref.read(settingsProvider).backendUrl).googleLogin(idToken);
      await ref.read(sharedPrefsProvider).setString(_tokenKey, token);
      state = state.copyWith(status: AuthStatus.signedIn, user: user, token: token, busy: false);
    } on ApiException catch (e) {
      state = state.copyWith(busy: false, error: e.message);
    }
  }

  String _friendly(Object e) {
    if (e is GoogleSignInException && e.code == GoogleSignInExceptionCode.canceled) return '';
    return 'Sign in nahi ho paaya: $e';
  }

  /// The server rejected our token (expired, or the user was removed).
  Future<void> sessionExpired() async {
    if (state.status != AuthStatus.signedIn || state.token == null) return;
    await signOut(message: 'Session khatam ho gaya. Dobara sign in karo.');
  }

  Future<void> signOut({String? message}) async {
    await ref.read(sharedPrefsProvider).remove(_tokenKey);
    try {
      if (_googleReady) await GoogleSignIn.instance.signOut();
    } catch (_) {}
    ref.read(sessionProvider.notifier).newChat();
    state = AuthState(status: AuthStatus.signedOut, googleClientId: state.googleClientId, error: message);
  }

  void retry() {
    state = const AuthState();
    _start(ref.read(settingsProvider).backendUrl);
  }
}

final authProvider = NotifierProvider<AuthNotifier, AuthState>(AuthNotifier.new);
