import 'dart:async';
import 'dart:convert';

import 'package:eduhinglish_app/api.dart';
import 'package:eduhinglish_app/auth.dart';
import 'package:eduhinglish_app/models.dart';
import 'package:eduhinglish_app/main.dart';
import 'package:eduhinglish_app/state.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

String sse(Map<String, dynamic> e) => 'data: ${jsonEncode(e)}\n\n';

Map<String, dynamic> msg(String id, String role, String content, [List<Map<String, dynamic>> sources = const []]) => {
      'id': id, 'chat_id': 'c1', 'role': role, 'content': content, 'sources': sources, 'feedback': 0, 'created_at': 1.0,
    };

/// A fake backend. [failStream] makes /api/chat/stream return a 503.
MockClient fakeBackend({bool failStream = false, List<String>? log}) {
  return MockClient.streaming((request, body) async {
    final path = request.url.path;
    log?.add('${request.method} $path');
    http.StreamedResponse json(Object data, [int status = 200]) => http.StreamedResponse(
        Stream.value(utf8.encode(jsonEncode(data))), status, headers: {'content-type': 'application/json'});

    if (path == '/api/chats') return json([]);
    if (path == '/api/chat/stream') {
      if (failStream) return json({'detail': 'Model server unreachable'}, 503);
      final chat = {'id': 'c1', 'title': 'Newton ka law', 'language': 'hinglish', 'created_at': 1.0, 'updated_at': 1.0};
      final events = [
        sse({'type': 'status', 'stage': 'retrieve'}),
        sse({'type': 'meta', 'intent': 'definition', 'sources': [{'chapter': 'Force and Laws of Motion', 'score': 0.8, 'text_preview': 'F = ma'}]}),
        sse({'type': 'token', 'text': 'F = '}),
        sse({'type': 'token', 'text': 'ma hota hai'}),
        sse({'type': 'done', 'chat': chat, 'user_message': msg('u1', 'user', 'Newton ka law?'), 'assistant_message': msg('a1', 'assistant', 'F = ma hota hai', [{'chapter': 'Force and Laws of Motion', 'score': 0.8, 'text_preview': 'F = ma'}])}),
      ];
      return http.StreamedResponse(Stream.fromIterable(events.map(utf8.encode)), 200, headers: {'content-type': 'text/event-stream'});
    }
    return json({'detail': 'not found'}, 404);
  });
}

/// Starts signed in, without touching the network or Google.
class _SignedInAuth extends AuthNotifier {
  @override
  AuthState build() => const AuthState(status: AuthStatus.signedIn, user: AppUser(id: 'u', name: 'Asha'));
}

Future<void> pumpApp(WidgetTester tester, MockClient client, {Size size = const Size(1200, 900)}) async {
  SharedPreferences.setMockInitialValues({});
  final prefs = await SharedPreferences.getInstance();
  tester.view.physicalSize = size;
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(ProviderScope(
    retry: (_, _) => null,
    overrides: [
      sharedPrefsProvider.overrideWithValue(prefs),
      authProvider.overrideWith(_SignedInAuth.new),
      apiProvider.overrideWith((ref) => ApiClient('http://test', client: client)),
    ],
    child: const EduHinglishApp(),
  ));
  await tester.pumpAndSettle();
}

void main() {
  testWidgets('welcome screen shows suggestions', (tester) async {
    await pumpApp(tester, fakeBackend());
    expect(find.text('Namaste! 👋'), findsOneWidget);
    expect(find.text('Newton ka third law samjhao'), findsOneWidget);
  });

  testWidgets('asking a question streams the answer and shows sources', (tester) async {
    await pumpApp(tester, fakeBackend());
    await tester.enterText(find.byType(TextField), 'Newton ka law?');
    await tester.pump();
    await tester.tap(find.byTooltip('Send'));
    await tester.pumpAndSettle();
    expect(find.text('Newton ka law?'), findsWidgets); // user bubble (+ sidebar title)
    expect(find.textContaining('F = ma hota hai'), findsOneWidget);
    expect(find.textContaining('NCERT sources (1)'), findsOneWidget);
    await tester.tap(find.textContaining('NCERT sources (1)'));
    await tester.pumpAndSettle();
    expect(find.textContaining('Force and Laws of Motion'), findsOneWidget);
  });

  testWidgets('a backend failure shows a retry banner and keeps the question', (tester) async {
    await pumpApp(tester, fakeBackend(failStream: true));
    await tester.enterText(find.byType(TextField), 'Newton ka law?');
    await tester.pump();
    await tester.tap(find.byTooltip('Send'));
    await tester.pumpAndSettle();
    expect(find.text('Model server unreachable'), findsOneWidget);
    expect(find.text('Retry'), findsOneWidget);
    expect(find.text('Newton ka law?'), findsOneWidget);
  });

  testWidgets('narrow screens use a drawer', (tester) async {
    await pumpApp(tester, fakeBackend(), size: const Size(420, 800));
    expect(find.byIcon(Icons.menu), findsOneWidget);
    await tester.tap(find.byIcon(Icons.menu));
    await tester.pumpAndSettle();
    expect(find.text('Naya chat'), findsOneWidget);
  });

  test('isAskable rejects punctuation-only input', () {
    expect(isAskable('??'), false);
    expect(isAskable('...'), false);
    expect(isAskable('a'), false);
    expect(isAskable('Ohm?'), true);
  });
}
