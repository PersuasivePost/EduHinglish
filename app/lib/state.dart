import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api.dart';
import 'auth.dart';
import 'config.dart';
import 'models.dart';

/// Same rule as the backend: at least two letters or digits, so "??" or "..." is never sent.
bool isAskable(String text) => RegExp(r'[\p{L}\p{N}]', unicode: true).allMatches(text).length >= 2;

// ── Settings (persisted) ─────────────────────────────────────────────────────

final sharedPrefsProvider = Provider<SharedPreferences>((ref) => throw UnimplementedError('override in main()'));

class Settings {
  const Settings({
    required this.backendUrl,
    this.themeMode = ThemeMode.system,
    this.subject,
    this.classNum,
    this.language = 'hinglish',
  });

  final String backendUrl;
  final ThemeMode themeMode;
  final String? subject; // null = all subjects
  final String? classNum; // null = both classes
  final String language;

  Settings copyWith({
    String? backendUrl,
    ThemeMode? themeMode,
    String? subject,
    bool clearSubject = false,
    String? classNum,
    bool clearClass = false,
    String? language,
  }) =>
      Settings(
        backendUrl: backendUrl ?? this.backendUrl,
        themeMode: themeMode ?? this.themeMode,
        subject: clearSubject ? null : (subject ?? this.subject),
        classNum: clearClass ? null : (classNum ?? this.classNum),
        language: language ?? this.language,
      );
}

class SettingsNotifier extends Notifier<Settings> {
  @override
  Settings build() {
    final p = ref.watch(sharedPrefsProvider);
    return Settings(
      // Only a URL typed in Settings is remembered; otherwise the build's default applies (BACKEND_URL / Render).
      backendUrl: p.getString('backendUrlCustom') ?? kDefaultBackendUrl,
      themeMode: ThemeMode.values.byName(p.getString('themeMode') ?? 'system'),
      subject: p.getString('subject'),
      classNum: p.getString('classNum'),
      language: p.getString('language') ?? 'hinglish',
    );
  }

  void _save(Settings s) {
    state = s;
    final p = ref.read(sharedPrefsProvider);
    p.setString('themeMode', s.themeMode.name);
    s.subject == null ? p.remove('subject') : p.setString('subject', s.subject!);
    s.classNum == null ? p.remove('classNum') : p.setString('classNum', s.classNum!);
    p.setString('language', s.language);
  }

  void setBackendUrl(String url) {
    final p = ref.read(sharedPrefsProvider);
    final custom = url.trim();
    if (custom.isEmpty || custom == kDefaultBackendUrl) {
      p.remove('backendUrlCustom');
      _save(state.copyWith(backendUrl: kDefaultBackendUrl));
    } else {
      p.setString('backendUrlCustom', custom);
      _save(state.copyWith(backendUrl: custom));
    }
  }
  void setThemeMode(ThemeMode m) => _save(state.copyWith(themeMode: m));
  void setSubject(String? s) => _save(s == null ? state.copyWith(clearSubject: true) : state.copyWith(subject: s));
  void setClass(String? c) => _save(c == null ? state.copyWith(clearClass: true) : state.copyWith(classNum: c));
  void setLanguage(String l) => _save(state.copyWith(language: l));
}

final settingsProvider = NotifierProvider<SettingsNotifier, Settings>(SettingsNotifier.new);

// ── API + chat list ──────────────────────────────────────────────────────────

final apiProvider = Provider<ApiClient>((ref) {
  final api = ApiClient(
    ref.watch(settingsProvider.select((s) => s.backendUrl)),
    token: ref.watch(authProvider.select((a) => a.token)),
    onUnauthorized: () => ref.read(authProvider.notifier).sessionExpired(),
  );
  ref.onDispose(api.close);
  return api;
});

class ChatsNotifier extends AsyncNotifier<List<Chat>> {
  @override
  Future<List<Chat>> build() => ref.watch(apiProvider).listChats();

  Future<void> refresh() async {
    try {
      state = AsyncData(await ref.read(apiProvider).listChats());
    } catch (e, st) {
      if (!state.hasValue) state = AsyncError(e, st);
    }
  }

  Future<void> delete(String id) async {
    await ref.read(apiProvider).deleteChat(id);
    state = AsyncData([...?state.value?.where((c) => c.id != id)]);
    if (ref.read(sessionProvider).chatId == id) ref.read(sessionProvider.notifier).newChat();
  }

  /// Publishes (or refreshes) the read-only link; returns the link's token.
  Future<String> share(String id) async {
    final chat = await ref.read(apiProvider).shareChat(id);
    state = AsyncData([for (final c in state.value ?? <Chat>[]) c.id == id ? chat : c]);
    return chat.shareToken!;
  }

  Future<void> unshare(String id) async {
    await ref.read(apiProvider).unshareChat(id);
    await refresh();
  }

  Future<void> rename(String id, String title) async {
    await ref.read(apiProvider).renameChat(id, title);
    state = AsyncData([for (final c in state.value ?? <Chat>[]) c.id == id ? c.copyWith(title: title) : c]);
  }
}

final chatsProvider = AsyncNotifierProvider<ChatsNotifier, List<Chat>>(ChatsNotifier.new);

// ── The open conversation ────────────────────────────────────────────────────

class Session {
  const Session({
    this.chatId,
    this.messages = const [],
    this.loading = false,
    this.sending = false,
    this.stage,
    this.error,
    this.lastText,
  });

  final String? chatId;
  final List<Message> messages; // oldest first
  final bool loading; // opening a saved chat
  final bool sending; // waiting for / receiving an answer
  final String? stage; // queued | rewrite | retrieve | generate
  final String? error;
  final String? lastText; // question to resend after an error

  Session copyWith({
    String? chatId,
    bool clearChat = false,
    List<Message>? messages,
    bool? loading,
    bool? sending,
    String? stage,
    bool clearStage = false,
    String? error,
    bool clearError = false,
    String? lastText,
  }) =>
      Session(
        chatId: clearChat ? null : (chatId ?? this.chatId),
        messages: messages ?? this.messages,
        loading: loading ?? this.loading,
        sending: sending ?? this.sending,
        stage: clearStage ? null : (stage ?? this.stage),
        error: clearError ? null : (error ?? this.error),
        lastText: lastText ?? this.lastText,
      );
}

class SessionNotifier extends Notifier<Session> {
  StreamSubscription<ChatEvent>? _sub;
  int _counter = 0;

  @override
  Session build() {
    ref.onDispose(() => _sub?.cancel());
    return const Session();
  }

  void newChat() {
    _sub?.cancel();
    state = const Session();
    // A new chat starts with "all subjects"; otherwise the previous chat's subject
    // silently filters retrieval (e.g. a Physics question answered from Chemistry).
    ref.read(settingsProvider.notifier).setSubject(null);
  }

  Future<void> open(String chatId) async {
    _sub?.cancel();
    state = Session(chatId: chatId, loading: true);
    try {
      final (chat, messages) = await ref.read(apiProvider).getChat(chatId);
      if (state.chatId != chatId) return; // user switched chats meanwhile
      final s = ref.read(settingsProvider.notifier);
      s.setSubject(chat.subject);
      s.setClass(chat.classNum);
      s.setLanguage(chat.language);
      state = Session(chatId: chatId, messages: messages);
    } on ApiException catch (e) {
      if (state.chatId == chatId) state = Session(chatId: chatId, error: e.message);
    }
  }

  /// Sends a question and streams the answer into a placeholder assistant message.
  void send(String text) {
    text = text.trim();
    if (!isAskable(text) || state.sending) return;
    final n = ++_counter;
    final userMsg = Message(id: 'local-u-$n', role: 'user', content: text);
    final botMsg = Message(id: 'local-a-$n', role: 'assistant', content: '', streaming: true);
    state = state.copyWith(
        messages: [...state.messages, userMsg, botMsg], sending: true, clearError: true, clearStage: true, lastText: text);
    _listen(text, botMsg.id, userMsg.id);
  }

  /// Retries the last question after an error or a stop.
  void retry() {
    final text = state.lastText;
    if (text == null || state.sending) return;
    // drop the unanswered user message; send() adds it again
    final msgs = [...state.messages];
    if (msgs.isNotEmpty && msgs.last.isUser) msgs.removeLast();
    state = state.copyWith(messages: msgs);
    send(text);
  }

  void stop() {
    if (!state.sending) return;
    _sub?.cancel();
    _fail('Rok diya gaya. Dobara bhejne ke liye Retry dabao.');
  }

  void _listen(String text, String botId, String userId) {
    final s = ref.read(settingsProvider);
    final stream = ref.read(apiProvider).streamChat(
          message: text,
          chatId: state.chatId,
          subject: s.subject,
          classNum: s.classNum,
          language: s.language,
        );
    _sub = stream.listen(
      (e) => _onEvent(e, botId, userId),
      onError: (Object e) => _fail(e is ApiException ? e.message : 'Kuch gadbad ho gayi: $e'),
      onDone: () {
        if (state.sending) _fail('Jawab poora aane se pehle connection band ho gaya.');
      },
    );
  }

  void _replace(String id, Message Function(Message) f) {
    state = state.copyWith(messages: [for (final m in state.messages) m.id == id ? f(m) : m]);
  }

  void _onEvent(ChatEvent e, String botId, String userId) {
    switch (e) {
      case StatusEvent():
        state = state.copyWith(stage: e.stage);
      case MetaEvent():
        _replace(botId, (m) => m.copyWith(sources: e.sources, intent: e.intent));
      case TokenEvent():
        _replace(botId, (m) => m.copyWith(content: m.content + e.text));
      case DoneEvent():
        final msgs = [
          for (final m in state.messages)
            if (m.id == userId) e.userMessage else if (m.id == botId) e.assistantMessage else m
        ];
        state = state.copyWith(chatId: e.chat.id, messages: msgs, sending: false, clearStage: true);
        ref.read(chatsProvider.notifier).refresh();
      case ErrorEvent():
        _fail(e.detail);
    }
  }

  void _fail(String message) {
    final msgs = state.messages
        .where((m) => !m.streaming)
        .map((m) => m.isUser && m.id.startsWith('local-u-') ? m.copyWith(failed: true) : m)
        .toList();
    state = state.copyWith(messages: msgs, sending: false, clearStage: true, error: message);
  }

  Future<void> setFeedback(Message m, int value) async {
    final v = m.feedback == value ? 0 : value; // tapping again clears it
    _replace(m.id, (x) => x.copyWith(feedback: v));
    try {
      await ref.read(apiProvider).setFeedback(m.id, v);
    } on ApiException {
      _replace(m.id, (x) => x.copyWith(feedback: m.feedback)); // roll back
    }
  }
}

final sessionProvider = NotifierProvider<SessionNotifier, Session>(SessionNotifier.new);
