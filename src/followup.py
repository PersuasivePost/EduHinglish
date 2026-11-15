"""
Follow-up handling for multi-turn chats.

The Gemma adapter was trained on single-turn prompts (context + one question), so we
never feed it the chat history. Instead a follow-up like "aur iska formula?" is first
rewritten into a standalone question ("Newton ke second law ka formula kya hai?").
Retrieval, intent detection and generation then see a normal single-turn question.

This module is pure Python (no torch), so it can be tested anywhere.
"""

import re

MAX_HISTORY_MESSAGES = 6      # last 3 exchanges are enough to resolve "it / this / iska"
MAX_HISTORY_CHARS = 300       # per message; answers are long and only the topic matters
MAX_REWRITE_CHARS = 300

def trim_history(history: list[dict]) -> list[dict]:
    """Keep the last few non-empty turns, with long answers shortened."""
    turns = []
    for m in history or []:
        content = " ".join((m.get("content") or "").split())
        if content and m.get("role") in ("user", "assistant"):
            if len(content) > MAX_HISTORY_CHARS:
                content = content[:MAX_HISTORY_CHARS].rstrip() + "..."
            turns.append({"role": m["role"], "content": content})
    return turns[-MAX_HISTORY_MESSAGES:]


def is_followup(query: str, history: list[dict]) -> bool:
    """Any chat with earlier turns gets the rewrite step, like ChatGPT reading the whole thread.

    The rewrite prompt tells the model to return a self-contained question unchanged,
    so standalone questions pass through. Cost: one short generation (~1-3 s) per follow-up turn.
    """
    return bool(trim_history(history))


def build_rewrite_prompt(history: list[dict], query: str) -> str:
    lines = []
    for m in trim_history(history):
        who = "Student" if m["role"] == "user" else "Tutor"
        lines.append(f"{who}: {m['content']}")
    convo = "\n".join(lines)
    return (
        "Rewrite the student's last question as one standalone question, using the conversation "
        "to fill in what words like 'it', 'this', 'iska', 'uska' refer to. "
        "Keep the same language and style (Hinglish stays Hinglish). "
        "Do not answer it. If it is already clear on its own, repeat it unchanged. "
        "Output only the question.\n\n"
        f"Conversation:\n{convo}\n\n"
        f"Last question: {query}\n"
        "Standalone question:"
    )


def clean_rewrite(raw: str, query: str = ""):
    """Return the model's rewrite if it looks sane, otherwise None.

    A rewrite far longer than the question is probably an answer, not a question.
    """
    text = (raw or "").strip().splitlines()[0].strip() if (raw or "").strip() else ""
    text = re.sub(r"^(standalone question|question)\s*:\s*", "", text, flags=re.I).strip(" \"'`*")
    if len(text) < 3 or len(text) > MAX_REWRITE_CHARS or (query and len(text) > 6 * len(query) + 120):
        return None
    return text


def heuristic_rewrite(query: str, history: list[dict]) -> str:
    """No-model fallback: attach the previous student question as context."""
    prev = next((m["content"] for m in reversed(trim_history(history)) if m["role"] == "user"), "")
    return f"{prev} {query}".strip() if prev else query
