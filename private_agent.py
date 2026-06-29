"""
private_agent.py - A private, local tool-calling agent.

Mirrors the GB10 build report's architecture, scaled to Apple Silicon:

    open model (Qwen3 8B / Granite 3.3 8B)
        -> local OpenAI-compatible endpoint (Ollama, 127.0.0.1:11434)
            -> agent loop (goal -> pick tool -> act -> report)

Nothing leaves the machine by default. The serving layer is localhost-bound, the
model runs on your own GPU (Metal), and the filesystem tools are sandboxed to a
single directory so the agent cannot wander.

Tools:
  list_dir, read_file, file_info  - always on, read-only.
  retrieve                        - always on, queries the local RAG index (rag.py).
  write_file                      - opt-in via --allow-write (sandboxed writes).
  fetch_url                       - opt-in via --allow-web (BREAKS sovereignty: leaves the box).

Usage:
    uv run private_agent.py --task inventory --dir ./sandbox
    uv run private_agent.py --goal "Who is the architect? Use retrieve."
    uv run private_agent.py --allow-write --goal "Write a one-line summary of people.csv to summary.txt"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

from openai import OpenAI

import rag

# --- Serving layer config (the vLLM-equivalent endpoint) -----------------------
# Ollama exposes an OpenAI-compatible API. The api_key is required by the SDK but
# unused by Ollama. Override with env vars to point at any OpenAI-shaped endpoint.
# Default to qwen3:8b - it drives the multi-step tool loop reliably on Ollama.
# granite3.3:8b (the report's model) is faithful to the report but flaky here: its
# tool-call output parses inconsistently through Ollama. Swap via PRIVATE_LLM_MODEL.
BASE_URL = os.environ.get("PRIVATE_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.environ.get("PRIVATE_LLM_MODEL", "qwen3:8b")

client = OpenAI(base_url=BASE_URL, api_key="ollama")


# --- The sandbox: filesystem tools confined to one root directory --------------
class Sandbox:
    """Filesystem tools confined to `root`. No path escapes the root."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root).resolve()

    def _resolve(self, path: str) -> Path:
        # Resolve the requested path and refuse anything outside the sandbox root.
        target = (self.root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
        if self.root != target and self.root not in target.parents:
            raise PermissionError(f"path escapes sandbox root: {path}")
        return target

    def list_dir(self, path: str = ".") -> str:
        target = self._resolve(path)
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {path}")
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return "\n".join(entries) if entries else "(empty directory)"

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"not a file: {path}")
        # Surface an honest signal for empty files instead of a silent "".
        text = target.read_text(encoding="utf-8", errors="replace")
        return text if text else "(file is empty - 0 bytes)"

    def file_info(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"no such path: {path}")
        st = target.stat()
        kind = "directory" if target.is_dir() else "file"
        return json.dumps({"path": path, "type": kind, "size_bytes": st.st_size})

    def write_file(self, path: str, content: str) -> str:
        # Opt-in (--allow-write). Still confined to the sandbox root.
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content.encode('utf-8'))} bytes to {path}"


def fetch_url(url: str) -> str:
    """Opt-in (--allow-web). NOTE: this leaves the box - it breaks the sovereignty story."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")
    req = urllib.request.Request(url, headers={"User-Agent": "private-agent/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 - opt-in only
        body = resp.read(200_000).decode("utf-8", errors="replace")
    text = re.sub(r"<[^>]+>", " ", body)  # crude tag strip
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000] + (" ...(truncated)" if len(text) > 2000 else "")


# --- Tool schemas. Capabilities are assembled per-run based on flags. ----------
def _schema(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}

_PATH = {"path": {"type": "string", "description": "Path, relative to the sandbox root."}}

ALL_SCHEMAS = {
    "list_dir": _schema("list_dir", "List the names of files and subdirectories in a directory.", _PATH, ["path"]),
    "read_file": _schema("read_file", "Read and return the full UTF-8 text contents of a file.", _PATH, ["path"]),
    "file_info": _schema("file_info", "Return the exact type and size in bytes of a file or directory. Use this for sizes - never guess.", _PATH, ["path"]),
    "retrieve": _schema("retrieve", "Search the local knowledge base for facts. Use this for any factual question instead of answering from memory.",
                        {"query": {"type": "string", "description": "What to look up."}}, ["query"]),
    "write_file": _schema("write_file", "Write UTF-8 text to a file in the sandbox (creates or overwrites).",
                          {**_PATH, "content": {"type": "string", "description": "Text to write."}}, ["path", "content"]),
    "fetch_url": _schema("fetch_url", "Fetch a web page and return its visible text. Leaves the local machine.",
                         {"url": {"type": "string", "description": "http(s) URL to fetch."}}, ["url"]),
}

# Behavioral guidance lives in the USER turn, not a system message. Granite's chat
# template handles tool-calling far more reliably this way - a heavy system prompt
# suppresses tool calls and the model hallucinates an answer instead. (Discovered the
# hard way; it mirrors the report's vLLM tool-call-parser mismatch.)
RULES = (
    "\n\nWork by calling the provided tools on REAL data - do not answer factual questions "
    "from memory; use retrieve. Copy filenames EXACTLY as list_dir returns them. Use "
    "size_bytes from file_info for sizes; never estimate. If a call fails or a file is empty, "
    "say so honestly - never invent contents. When done, reply with a short plain-text answer."
)


def run_agent(goal, sandbox, *, allow_write=False, allow_web=False, force_first=None,
              max_steps=12, verbose=True):
    """Drive the goal -> tool -> act -> report loop against the local model."""
    dispatch = {
        "list_dir": sandbox.list_dir,
        "read_file": sandbox.read_file,
        "file_info": sandbox.file_info,
        "retrieve": lambda query: rag.retrieve(query),
    }
    if allow_write:
        dispatch["write_file"] = sandbox.write_file
    if allow_web:
        dispatch["fetch_url"] = fetch_url

    tools = [ALL_SCHEMAS[name] for name in dispatch]
    allowed = {name: set(ALL_SCHEMAS[name]["function"]["parameters"]["properties"]) for name in dispatch}

    messages = [{"role": "user", "content": goal + RULES}]

    for step in range(1, max_steps + 1):
        # Optionally ground the first turn by forcing a specific opening call,
        # instead of trusting an 8B to start with a real tool call.
        tool_choice = (
            {"type": "function", "function": {"name": force_first}}
            if step == 1 and force_first else "auto"
        )
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, tool_choice=tool_choice, temperature=0
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            if verbose:
                print(f"\n=== agent finished in {step} model turn(s) ===")
            return msg.content or "(no final message)"

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                call_args = json.loads(raw_args)
                # Some models wrap args as {"arguments": {...}, "label": ...}; unwrap,
                # then keep only the parameters this tool actually accepts.
                if isinstance(call_args.get("arguments"), dict):
                    call_args = call_args["arguments"]
                call_args = {k: v for k, v in call_args.items() if k in allowed.get(name, set())}
                result = dispatch[name](**call_args)
            except Exception as exc:  # honest failure signal back to the model
                result = f"ERROR: {type(exc).__name__}: {exc}"
            if verbose:
                shown = result if len(result) <= 200 else result[:200] + " ...(truncated)"
                print(f"  [step {step}] {name}({raw_args}) -> {shown!r}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "(stopped: hit max_steps without a final answer)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Private local tool-calling agent.")
    parser.add_argument("--goal", help="Free-form goal for the agent.")
    parser.add_argument("--task", choices=["inventory"], help="Built-in calibration task.")
    parser.add_argument("--dir", default="./sandbox", help="Sandbox root (default: ./sandbox).")
    parser.add_argument("--allow-write", action="store_true", help="Enable the sandboxed write_file tool.")
    parser.add_argument("--allow-web", action="store_true", help="Enable fetch_url (leaves the box).")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step tool trace.")
    args = parser.parse_args()

    sandbox = Sandbox(args.dir)
    force_first = None

    if args.task == "inventory":
        goal = (
            "Inventory the sandbox directory. List every file, then for EACH file report: "
            "its exact filename, its size in bytes, and a one-line summary of its contents."
        )
        force_first = "list_dir"
    elif args.goal:
        goal = args.goal
    else:
        parser.error("provide either --goal or --task")

    caps = ["read", "retrieve"] + (["write"] if args.allow_write else []) + (["web"] if args.allow_web else [])
    print(f"model    : {MODEL}")
    print(f"endpoint : {BASE_URL}")
    print(f"sandbox  : {sandbox.root}")
    print(f"caps     : {', '.join(caps)}")
    print(f"goal     : {goal}\n")

    answer = run_agent(
        goal, sandbox, allow_write=args.allow_write, allow_web=args.allow_web,
        force_first=force_first, verbose=not args.quiet,
    )
    print("\n--- AGENT REPORT ---\n" + answer)


if __name__ == "__main__":
    main()
