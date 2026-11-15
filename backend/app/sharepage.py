"""The read-only page behind a share link (/s/<token>). Plain HTML: no login, no app, no model server."""

from html import escape

_CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#6b6b6b;--user:#f0f0f3;--line:#e6e6ea;--accent:#3f4a8a}
@media(prefers-color-scheme:dark){:root{--bg:#161618;--fg:#ececf0;--muted:#9a9aa3;--user:#26262b;--line:#2c2c32;--accent:#a9b4ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:20px;margin:0 0 4px}.meta{color:var(--muted);font-size:13px;margin-bottom:24px}
.msg{margin:18px 0;white-space:pre-wrap;word-wrap:break-word}
.user{background:var(--user);padding:10px 14px;border-radius:18px;margin-left:15%}
.bot{padding:0 2px}.who{font-size:12px;color:var(--muted);margin-bottom:2px}
.src{font-size:12px;color:var(--muted);margin-top:6px}
footer{margin-top:40px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
button{font:inherit;font-size:13px;background:none;color:var(--accent);border:1px solid var(--line);border-radius:8px;padding:4px 10px;cursor:pointer}
"""

_JS = """
document.getElementById('copy').onclick=function(){
  var t=[].map.call(document.querySelectorAll('.msg'),function(m){return m.dataset.who+': '+m.dataset.text}).join('\\n\\n');
  navigator.clipboard.writeText(t).then(function(){document.getElementById('copy').textContent='Copied'});
};
"""


def render_shared(chat, messages) -> str:
    if not chat:
        body = "<h1>Link nahi mila</h1><p class='meta'>This shared chat does not exist or was made private.</p>"
        title = "Not found"
    else:
        title = chat["title"]
        parts = [f"<h1>{escape(title)}</h1>",
                 "<div class='meta'>Shared EduHinglish chat · read-only "
                 "<button id='copy'>Copy chat</button></div>"]
        for m in messages:
            who = "Student" if m["role"] == "user" else "EduHinglish"
            text = escape(m["content"])
            cls = "user" if m["role"] == "user" else "bot"
            src = ""
            if m["role"] == "assistant" and m.get("sources"):
                chapters = sorted({s.get("chapter", "") for s in m["sources"] if s.get("chapter")})
                if chapters:
                    src = f"<div class='src'>NCERT: {escape(', '.join(chapters))}</div>"
            parts.append(f"<div class='msg {cls}' data-who='{who}' data-text=\"{escape(m['content'], quote=True)}\">"
                         f"<div class='who'>{who}</div>{text}{src}</div>")
        parts.append("<footer>EduHinglish — Hinglish AI tutor for NCERT Class 9–10. Answers can contain mistakes; "
                     "check with your textbook.</footer>")
        body = "".join(parts)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<meta name='robots' content='noindex'><title>{escape(title)} · EduHinglish</title>"
            f"<style>{_CSS}</style></head><body><main>{body}</main><script>{_JS}</script></body></html>")
