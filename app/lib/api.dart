import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'models.dart';

class AuthConfig {
  const AuthConfig({required this.enabled, this.googleClientId});
  final bool enabled;
  final String? googleClientId;
}

class ApiException implements Exception {
  ApiException(this.message, {this.status});
  final String message;
  final int? status;
  @override
  String toString() => message;
}

const _offlineHint =
    'Server se connect nahi ho paaya. Free server so raha ho sakta hai — 30-60 second baad dobara try karo.';

/// Talks to the EduHinglish backend (backend/app/main.py). The app never calls the model directly.
class ApiClient {
  ApiClient(String baseUrl, {http.Client? client, this.token, this.onUnauthorized})
      : baseUrl = baseUrl.trim().replaceAll(RegExp(r'/+$'), ''),
        _http = client ?? http.Client();

  final String baseUrl;
  final http.Client _http;

  /// Session token from /api/auth/google, sent on every call.
  final String? token;

  /// Called when the server says the session is no longer valid (401).
  final void Function()? onUnauthorized;

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  Map<String, String> _headers({bool json = false}) => {
        if (json) 'Content-Type': 'application/json',
        if (token != null) 'Authorization': 'Bearer $token',
      };

  /// Link anyone can open (no app, no login) to read a shared chat.
  String shareUrl(String shareToken) => '$baseUrl/s/$shareToken';

  Future<dynamic> _json(Future<http.Response> Function() call) async {
    try {
      final r = await call().timeout(const Duration(seconds: 90));
      if (r.statusCode == 401) onUnauthorized?.call();
      if (r.statusCode >= 400) throw ApiException(_detail(r.body, r.statusCode), status: r.statusCode);
      return r.body.isEmpty ? null : jsonDecode(utf8.decode(r.bodyBytes));
    } on ApiException {
      rethrow;
    } on TimeoutException {
      throw ApiException(_offlineHint);
    } on http.ClientException {
      throw ApiException(_offlineHint);
    }
  }

  String _detail(String body, int status) {
    try {
      final d = jsonDecode(body);
      if (d is Map && d['detail'] != null) return d['detail'].toString();
    } catch (_) {}
    return 'Server error ($status)';
  }

  Future<Map<String, dynamic>> health() async => Map<String, dynamic>.from(await _json(() => _http.get(_uri('/api/health'), headers: _headers())));

  Future<List<Chat>> listChats() async {
    final list = await _json(() => _http.get(_uri('/api/chats'), headers: _headers())) as List;
    return list.map((c) => Chat.fromJson(Map<String, dynamic>.from(c as Map))).toList();
  }

  Future<(Chat, List<Message>)> getChat(String id) async {
    final j = Map<String, dynamic>.from(await _json(() => _http.get(_uri('/api/chats/$id'), headers: _headers())));
    final msgs = (j['messages'] as List).map((m) => Message.fromJson(Map<String, dynamic>.from(m as Map))).toList();
    return (Chat.fromJson(j), msgs);
  }

  Future<void> deleteChat(String id) => _json(() => _http.delete(_uri('/api/chats/$id'), headers: _headers()));

  Future<void> renameChat(String id, String title) => _json(() => _http.patch(_uri('/api/chats/$id'),
      headers: _headers(json: true), body: jsonEncode({'title': title})));

  Future<void> setFeedback(String messageId, int value) => _json(() => _http.post(
      _uri('/api/messages/$messageId/feedback'),
      headers: _headers(json: true),
      body: jsonEncode({'value': value})));

  Future<AuthConfig> authConfig() async {
    final j = Map<String, dynamic>.from(await _json(() => _http.get(_uri('/api/auth/config'))));
    return AuthConfig(enabled: j['enabled'] == true, googleClientId: j['google_client_id'] as String?);
  }

  /// Exchanges a Google ID token for our session token.
  Future<(String, AppUser)> googleLogin(String idToken) async {
    final j = Map<String, dynamic>.from(await _json(() =>
        _http.post(_uri('/api/auth/google'), headers: _headers(json: true), body: jsonEncode({'id_token': idToken}))));
    return (j['token'] as String, AppUser.fromJson(Map<String, dynamic>.from(j['user'] as Map)));
  }

  Future<AppUser> me() async => AppUser.fromJson(Map<String, dynamic>.from(await _json(() => _http.get(_uri('/api/me'), headers: _headers()))));

  /// Publishes (or refreshes) the read-only link of a chat.
  Future<Chat> shareChat(String id) async => Chat.fromJson(
      Map<String, dynamic>.from(await _json(() => _http.post(_uri('/api/chats/$id/share'), headers: _headers()))));

  Future<void> unshareChat(String id) => _json(() => _http.delete(_uri('/api/chats/$id/share'), headers: _headers()));

  /// Streams one answer. Cancel the subscription to stop reading.
  Stream<ChatEvent> streamChat({
    required String message,
    String? chatId,
    String? subject,
    String? classNum,
    String language = 'hinglish',
  }) async* {
    final req = http.Request('POST', _uri('/api/chat/stream'))
      ..headers.addAll(_headers(json: true))
      ..headers['Accept'] = 'text/event-stream'
      ..body = jsonEncode({
        'message': message,
        'chat_id': ?chatId,
        'subject': ?subject,
        'class_num': ?classNum,
        'language': language,
      });
    http.StreamedResponse res;
    try {
      res = await _http.send(req).timeout(const Duration(seconds: 90));
    } on TimeoutException {
      yield ErrorEvent(503, _offlineHint);
      return;
    } on http.ClientException {
      yield ErrorEvent(503, _offlineHint);
      return;
    }
    if (res.statusCode == 401) onUnauthorized?.call();
    if (res.statusCode >= 400) {
      yield ErrorEvent(res.statusCode, _detail(await res.stream.bytesToString(), res.statusCode));
      return;
    }
    try {
      final lines = res.stream.transform(utf8.decoder).transform(const LineSplitter());
      await for (final line in lines) {
        if (!line.startsWith('data: ')) continue;
        final event = parseEvent(Map<String, dynamic>.from(jsonDecode(line.substring(6)) as Map));
        if (event != null) yield event;
      }
    } on http.ClientException {
      yield ErrorEvent(503, 'Connection beech mein toot gaya. Dobara try karo.');
    }
  }

  static ChatEvent? parseEvent(Map<String, dynamic> j) {
    switch (j['type']) {
      case 'status':
        return StatusEvent(j['stage'] as String);
      case 'meta':
        return MetaEvent(
          intent: j['intent'] as String?,
          sources: ((j['sources'] as List?) ?? const [])
              .map((s) => Source.fromJson(Map<String, dynamic>.from(s as Map)))
              .toList(),
        );
      case 'token':
        return TokenEvent(j['text'] as String);
      case 'done':
        return DoneEvent(
          chat: Chat.fromJson(Map<String, dynamic>.from(j['chat'] as Map)),
          userMessage: Message.fromJson(Map<String, dynamic>.from(j['user_message'] as Map)),
          assistantMessage: Message.fromJson(Map<String, dynamic>.from(j['assistant_message'] as Map)),
        );
      case 'error':
        return ErrorEvent((j['status'] as num?)?.toInt() ?? 502, (j['detail'] ?? 'Unknown error').toString());
    }
    return null;
  }

  void close() => _http.close();
}
