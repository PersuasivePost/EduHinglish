class Source {
  Source({required this.chapter, this.subject = '', this.classNum = '', this.score = 0, this.textPreview = ''});

  final String chapter;
  final String subject;
  final String classNum;
  final double score;
  final String textPreview;

  factory Source.fromJson(Map<String, dynamic> j) => Source(
        chapter: (j['chapter'] ?? 'Unknown').toString(),
        subject: (j['subject'] ?? '').toString(),
        classNum: (j['class_num'] ?? j['class'] ?? '').toString(),
        score: (j['score'] as num?)?.toDouble() ?? 0,
        textPreview: (j['text_preview'] ?? '').toString(),
      );
}

class AppUser {
  const AppUser({required this.id, this.name, this.email, this.picture});

  final String id;
  final String? name;
  final String? email;
  final String? picture;

  factory AppUser.fromJson(Map<String, dynamic> j) =>
      AppUser(id: j['id'] as String, name: j['name'] as String?, email: j['email'] as String?, picture: j['picture'] as String?);
}

class Chat {
  Chat({
    required this.id,
    required this.title,
    this.subject,
    this.classNum,
    this.language = 'hinglish',
    this.updatedAt = 0,
    this.shareToken,
  });

  final String id;
  final String title;
  final String? subject;
  final String? classNum;
  final String language;
  final double updatedAt;
  final String? shareToken; // set while the chat has a public read-only link

  Chat copyWith({String? title}) => Chat(
      id: id,
      title: title ?? this.title,
      subject: subject,
      classNum: classNum,
      language: language,
      updatedAt: updatedAt,
      shareToken: shareToken);

  factory Chat.fromJson(Map<String, dynamic> j) => Chat(
        id: j['id'] as String,
        title: j['title'] as String,
        subject: j['subject'] as String?,
        classNum: j['class_num'] as String?,
        language: (j['language'] as String?) ?? 'hinglish',
        updatedAt: (j['updated_at'] as num?)?.toDouble() ?? 0,
        shareToken: j['share_token'] as String?,
      );
}

class Message {
  Message({
    required this.id,
    required this.role,
    required this.content,
    this.sources = const [],
    this.intent,
    this.feedback = 0,
    this.streaming = false,
    this.failed = false,
  });

  final String id;
  final String role; // 'user' | 'assistant'
  final String content;
  final List<Source> sources;
  final String? intent;
  final int feedback; // 1 = 👍, -1 = 👎, 0 = none
  final bool streaming; // assistant answer still being written
  final bool failed; // user message that never got an answer

  bool get isUser => role == 'user';

  Message copyWith({String? content, List<Source>? sources, String? intent, int? feedback, bool? streaming, bool? failed}) =>
      Message(
        id: id,
        role: role,
        content: content ?? this.content,
        sources: sources ?? this.sources,
        intent: intent ?? this.intent,
        feedback: feedback ?? this.feedback,
        streaming: streaming ?? this.streaming,
        failed: failed ?? this.failed,
      );

  factory Message.fromJson(Map<String, dynamic> j) => Message(
        id: j['id'] as String,
        role: j['role'] as String,
        content: (j['content'] ?? '').toString(),
        intent: j['intent'] as String?,
        feedback: (j['feedback'] as num?)?.toInt() ?? 0,
        sources: ((j['sources'] as List?) ?? const [])
            .map((s) => Source.fromJson(Map<String, dynamic>.from(s as Map)))
            .toList(),
      );
}

/// Events of POST /api/chat/stream (Server-Sent Events).
sealed class ChatEvent {}

class StatusEvent extends ChatEvent {
  StatusEvent(this.stage);
  final String stage; // queued | rewrite | retrieve | generate
}

class MetaEvent extends ChatEvent {
  MetaEvent({this.intent, this.sources = const []});
  final String? intent;
  final List<Source> sources;
}

class TokenEvent extends ChatEvent {
  TokenEvent(this.text);
  final String text;
}

class DoneEvent extends ChatEvent {
  DoneEvent({required this.chat, required this.userMessage, required this.assistantMessage});
  final Chat chat;
  final Message userMessage;
  final Message assistantMessage;
}

class ErrorEvent extends ChatEvent {
  ErrorEvent(this.status, this.detail);
  final int status;
  final String detail;
}
