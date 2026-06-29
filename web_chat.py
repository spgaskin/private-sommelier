"""
web_chat.py - a local web front end to chat with the private wine assistant.

Retrieval-augmented chat over the wine knowledge base (kb_wine/ + X-Wines ->
wine_index.npz). The /chat endpoint STREAMS its progress over Server-Sent Events so
the user sees what's happening: searching the cellar, which wines were retrieved, then
the answer written live. Bound to 127.0.0.1 only - same sovereignty posture as the rest.

Run:   uv run web_chat.py      then open http://127.0.0.1:8080
"""

from __future__ import annotations

import json
import os
import re

# Point the RAG layer at the wine index before importing it.
os.environ.setdefault("PRIVATE_LLM_INDEX", "wine_index.npz")

from flask import Flask, Response, request

from openai import OpenAI

import rag

BASE_URL = os.environ.get("PRIVATE_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.environ.get("PRIVATE_LLM_MODEL", "qwen3:8b")
TOP_K = int(os.environ.get("WINE_TOP_K", "5"))

client = OpenAI(base_url=BASE_URL, api_key="ollama")

SYSTEM = (
    "You are a warm, knowledgeable sommelier. Answer the user's wine questions using the "
    "CONTEXT below, which is retrieved from a local wine knowledge base. Prefer the context "
    "over your own memory. If the context does not cover something, you may add general "
    "knowledge but say so - and never invent specifics like exact vintages, scores, or "
    "prices. Be concise, practical, and friendly. /no_think"
)

app = Flask(__name__)


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def visible_text(raw: str) -> str:
    """Hide <think> reasoning: drop completed blocks and any unclosed trailing one."""
    s = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cut = s.find("<think>")
    return s[:cut] if cut != -1 else s


def pretty_source(src: str) -> str:
    return src.split("x-wines: ", 1)[-1] if src.startswith("x-wines: ") else src


@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    history = data.get("history", [])
    user_msg = (data.get("message") or "").strip()

    def gen():
        if not user_msg:
            yield sse({"type": "done", "sources": []})
            return

        # Stage 1: retrieve grounding context.
        yield sse({"type": "status", "stage": "search", "text": "Searching the cellar…"})
        try:
            hits = rag.search(user_msg, k=TOP_K)
        except FileNotFoundError:
            yield sse({"type": "token", "text": "The wine index is missing. Run: "
                       "uv run rag.py ingest --dir ./kb_wine"})
            yield sse({"type": "done", "sources": []})
            return

        names = [pretty_source(s) for _, s, _ in hits]
        context = "\n\n".join(f"[{src}]\n{text}" for _, src, text in hits)
        sources = sorted({pretty_source(s) for _, s, _ in hits})
        # Stage 2: tell the user what was found.
        yield sse({"type": "status", "stage": "found",
                   "text": f"Found {len(hits)} matches · reading tasting notes",
                   "names": names})

        # Stage 3: stream the model's answer token by token.
        yield sse({"type": "status", "stage": "compose", "text": "Composing your answer…"})
        messages = (
            [{"role": "system", "content": SYSTEM + "\n\nCONTEXT:\n" + context}]
            + history[-8:]
            + [{"role": "user", "content": user_msg}]
        )
        try:
            stream = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0.3, stream=True
            )
            raw, sent = "", 0
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                raw += delta
                vis = visible_text(raw)
                if len(vis) > sent:
                    yield sse({"type": "token", "text": vis[sent:]})
                    sent = len(vis)
        except Exception as exc:
            yield sse({"type": "token", "text": f"\n[error talking to the model: {exc}]"})
        yield sse({"type": "done", "sources": sources})

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return PAGE


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Private Sommelier</title>
<style>
  :root{--wine:#6b1f2a;--wine2:#8c2b3a;--paper:#faf6f2;--ink:#2a1a1d;--muted:#7a6a6c;--line:#e7d8d2}
  *{box-sizing:border-box}
  body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    background:var(--paper);color:var(--ink);height:100vh;display:flex;flex-direction:column}
  header{background:linear-gradient(135deg,var(--wine),var(--wine2));color:#fff;padding:16px 22px}
  header h1{margin:0;font-size:19px;letter-spacing:-.01em}
  header p{margin:3px 0 0;font-size:12.5px;opacity:.82}
  #chat{flex:1;overflow-y:auto;padding:22px;max-width:820px;width:100%;margin:0 auto}
  .msg{margin:0 0 16px;display:flex;gap:10px;align-items:flex-start}
  .msg .who{font-size:12px;font-weight:700;width:64px;flex:none;padding-top:4px;color:var(--muted)}
  .msg.user .who{color:var(--wine)}
  .bubble{background:#fff;border:1px solid var(--line);border-radius:12px;padding:11px 15px;
    line-height:1.55;font-size:15px;white-space:pre-wrap;max-width:680px}
  .msg.user .bubble{background:#f3e7e3;border-color:#e7cfc8}
  /* live status while the somm works */
  .status{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted);font-style:italic}
  .status .spin{width:11px;height:11px;border:2px solid var(--line);border-top-color:var(--wine);
    border-radius:50%;animation:spin .8s linear infinite;flex:none}
  @keyframes spin{to{transform:rotate(360deg)}}
  .chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px}
  .chip{font-size:11px;background:#f3e7e3;color:var(--wine);border:1px solid #e7cfc8;
    border-radius:999px;padding:2px 9px}
  .answer{margin-top:2px}
  .cursor{display:inline-block;width:7px;height:15px;background:var(--wine);margin-left:1px;
    vertical-align:-2px;animation:blink 1s step-end infinite}
  @keyframes blink{50%{opacity:0}}
  .src{font-size:11.5px;color:var(--muted);margin-top:8px}
  .src b{color:var(--wine)}
  footer{border-top:1px solid var(--line);background:#fff;padding:12px}
  form{display:flex;gap:10px;max-width:820px;margin:0 auto}
  input{flex:1;border:1px solid var(--line);border-radius:10px;padding:12px 14px;font-size:15px;outline:none}
  input:focus{border-color:var(--wine)}
  button{background:var(--wine);color:#fff;border:0;border-radius:10px;padding:0 20px;font-size:15px;
    font-weight:600;cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
  .hint{max-width:820px;margin:8px auto 0;font-size:11.5px;color:var(--muted)}
</style></head><body>
<header>
  <h1>🍷 Private Sommelier</h1>
  <p>Local model · wine knowledge in retrieval · nothing leaves your Mac</p>
</header>
<div id="chat"></div>
<footer>
  <form id="f">
    <input id="q" autocomplete="off" placeholder="Ask about a region, grape, pairing, serving temp…" autofocus>
    <button id="send">Ask</button>
  </form>
  <div class="hint">e.g. "Recommend a highly-rated Spanish red" · "What pairs with lamb?" · "How is Champagne made?"</div>
</footer>
<script>
const chat=document.getElementById('chat'),form=document.getElementById('f'),
      q=document.getElementById('q'),send=document.getElementById('send');
const history=[];
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function add(role){const m=document.createElement('div');m.className='msg '+role;
  m.innerHTML='<div class="who">'+(role==='user'?'You':'Somm')+'</div><div class="bubble"></div>';
  chat.appendChild(m);chat.scrollTop=chat.scrollHeight;return m.querySelector('.bubble');}

add('bot').textContent='Pour yourself a glass and ask me anything about wine — regions, grapes, pairings, serving, or a recommendation from the cellar.';

form.onsubmit=async e=>{e.preventDefault();const text=q.value.trim();if(!text)return;
  add('user').textContent=text;q.value='';send.disabled=true;
  const b=add('bot');
  // bubble has: a live status line, an answer area, a sources line
  b.innerHTML='<div class="status"><span class="spin"></span><span class="stxt">Uncorking…</span></div>'
    +'<div class="chips"></div><div class="answer"></div><div class="src"></div>';
  const stxt=b.querySelector('.stxt'),chips=b.querySelector('.chips'),
        ans=b.querySelector('.answer'),src=b.querySelector('.src');
  let answer='',streaming=false;
  ans.innerHTML='';
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({message:text,history})});
    const reader=r.body.getReader(),dec=new TextDecoder();let buf='';
    while(true){const{value,done}=await reader.read();if(done)break;
      buf+=dec.decode(value,{stream:true});
      const parts=buf.split('\\n\\n');buf=parts.pop();
      for(const p of parts){const line=p.replace(/^data: /,'').trim();if(!line)continue;
        const ev=JSON.parse(line);
        if(ev.type==='status'){stxt.textContent=ev.text;
          if(ev.names){chips.innerHTML=ev.names.map(n=>'<span class="chip">'+esc(n)+'</span>').join('');}}
        else if(ev.type==='token'){
          if(!streaming){streaming=true;stxt.parentElement.remove();}
          answer+=ev.text;ans.innerHTML=esc(answer)+'<span class="cursor"></span>';
          chat.scrollTop=chat.scrollHeight;}
        else if(ev.type==='done'){
          ans.innerHTML=esc(answer);
          if(ev.sources&&ev.sources.length){src.innerHTML='<b>sources:</b> '+ev.sources.map(esc).join(', ');}
          if(!streaming&&b.querySelector('.status'))b.querySelector('.status').remove();
          history.push({role:'user',content:text});history.push({role:'assistant',content:answer});}
      }}
  }catch(err){ans.textContent='Error: '+err;}
  send.disabled=false;q.focus();chat.scrollTop=chat.scrollHeight;};
</script></body></html>"""


if __name__ == "__main__":
    print(f"Private Sommelier  ->  http://127.0.0.1:8080   (model: {MODEL})")
    app.run(host="127.0.0.1", port=8080, debug=False, threaded=True)
