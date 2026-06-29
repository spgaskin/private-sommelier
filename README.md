# Private LLM — local agent stack (Mac edition)

A working, self-contained private LLM + agent, built to mirror the architecture in
`gb10-private-llm-report-sanitized (1).html` — but scaled from a GB10 / DGX Spark box
down to **Apple Silicon (M5, 32 GB)**. Nothing leaves the machine.

```
open model (Qwen3 8B / Granite 3.3 8B)
   -> local OpenAI-compatible endpoint (Ollama, 127.0.0.1:11434)
      -> tool-calling agent loop  (goal -> pick tool -> act -> report)
```

## How this maps to the report

| Report (GB10 box)                     | This build (your Mac)                          |
| ------------------------------------- | ---------------------------------------------- |
| GB10 Grace Blackwell, 128 GB unified  | Apple M5, 32 GB unified                         |
| Granite 4.1 8B (open weights)         | Qwen3 8B (default) / Granite 3.3 8B (faithful)  |
| vLLM, OpenAI endpoint `:8000`         | **Ollama**, OpenAI endpoint `:11434`            |
| Hermes Agent `:9119`                  | `private_agent.py` — a minimal tool-call loop   |
| Localhost-bound + SSH tunnel only     | Ollama binds `127.0.0.1` by default             |
| Fine-tune for behavior, RAG for facts | (next phase — not built here)                   |

Same thesis as the report: **sovereignty by construction**. The model runs on your own
GPU (Metal), the endpoint is localhost-only, and the agent's tools are read-only and
sandboxed to a single directory so it cannot wander the filesystem.

## Prerequisites (already set up in this session)

- **Ollama** — installed via Homebrew, runs as a localhost service:
  `brew services start ollama` (endpoint at `http://127.0.0.1:11434`)
- **Models** — pulled into `~/.ollama` (separate from this folder):
  `ollama pull qwen3:8b` and `ollama pull granite3.3:8b`
- **Python** — managed by `uv` (Python 3.11, `openai` SDK). No global installs.

**New here:** a printable **[How-To-Use.pdf](How-To-Use.pdf)** covers all of this in one page-and-a-bit.

## Run it

```bash
# the report's calibration task: inventory a folder, summarize each file
uv run private_agent.py --task inventory --dir ./sandbox

# a factual question - answered from the knowledge base via the retrieve tool
uv run private_agent.py --goal "What port is billing on? Use retrieve."

# let the agent write a file (opt-in, sandboxed)
uv run private_agent.py --allow-write --goal "Read people.csv and write a summary to summary.txt."

# swap the model (e.g. run the report-faithful Granite to watch it struggle)
PRIVATE_LLM_MODEL=granite3.3:8b uv run private_agent.py --task inventory --dir ./sandbox
```

## Tools the agent has

| Tool | Default | Notes |
| ---- | ------- | ----- |
| `list_dir`, `read_file`, `file_info` | on | read-only, sandboxed |
| `retrieve` | on | searches the local RAG index (`kb/`) — facts, not memory |
| `write_file` | `--allow-write` | sandboxed writes only |
| `fetch_url` | `--allow-web` | **leaves the box** — breaks sovereignty, opt-in |

## Knowledge base (RAG) — facts live in retrieval, not weights

```bash
uv run rag.py ingest --dir ./kb           # (re)build the index after editing kb/
uv run rag.py query "who is the architect?"
```

Edit a file in `kb/`, re-ingest, and the fact is updated — no retraining. This is the
report's "behavior in weights, facts in retrieval" split, made concrete.

## Wine Sommelier — web app over real wine data

A local web front end that chats over a wine knowledge base, RAG-grounded. Localhost-only.

```bash
uv run web_chat.py          # then open http://127.0.0.1:8080
```

Data layers in the `wine_index.npz` index:
- `kb_wine/` — a curated, accurate reference base (regions, grapes, classifications, pairing).
- **X-Wines** — 100K real wines + averages from 21M user ratings, an openly-licensed dataset
  ([repo](https://github.com/rogerioxavier/X-Wines), [paper](https://www.mdpi.com/2504-2289/7/1/20)).
  Loaded with `load_xwines.py`:

```bash
uv run load_xwines.py --limit 4000 --order ratings   # most-reviewed subset (fast)
uv run load_xwines.py --limit 0                       # all 100K wines (slow, runs in background)
```

The loader builds one retrieval doc per wine (type, region, winery, grapes, body, acidity,
ABV, food pairings, average star rating) and re-folds the curated base, so the assistant
answers from both reference knowledge and real bottles. Data note: X-Wines is used under its
open research license; scraped/paywalled sources (Wine Spectator, wine.com, Vivino) are
deliberately **not** used.

## Fine-tuning (scaffolded — the report's "next phase")

QLoRA via MLX on Apple Silicon, tuning *behavior* (cite sources, refuse to guess) while
facts stay in RAG:

```bash
cd finetune && ./run_finetune.sh          # writes LoRA adapters to ./adapters
```

See [finetune/README.md](finetune/README.md) for fusing the adapters and serving the
tuned model through the same local endpoint.

## What the build surfaced (matches the report's findings)

The report's headline finding was that **an 8B driving the loop turns a failed/absent
tool call into a confident false claim** — the fragility is fidelity, not reasoning.
Reproduced here, twice over:

1. **Granite 3.3 8B is unreliable at agentic tool-calling through Ollama.** It would
   write Python code instead of calling tools, print tool calls as text, or hallucinate
   an entire directory listing without calling anything. A heavy *system* prompt made it
   worse — moving guidance into the *user* turn was the fix that restored tool calls
   (the Ollama analogue of the report's `--tool-call-parser hermes` mismatch). Even so it
   stayed flaky, so the default model here is Qwen3 8B.

2. **Qwen3 8B drives the loop reliably** — exact filenames preserved (every hyphen
   intact, unlike the report's Granite), empty files honestly reported as empty. **But**
   on a run where it skipped `file_info` for two files, it *fabricated their byte sizes*
   rather than admitting it hadn't checked. Same lesson as the report: anything an 8B
   produces unsupervised needs verification.

**Takeaway, same as the report:** keep facts in tools/retrieval, verify tool outputs,
and don't trust an 8B's unsupervised claims. The honest-failure design here — surfacing
errors and `(file is empty - 0 bytes)` instead of silent `""` — is the verification seam.

## Files

- `private_agent.py` — the agent: serving config, sandboxed tools, the loop.
- `rag.py` — tiny local RAG (Ollama embeddings + numpy cosine search, one `.npz` index).
- `kb/` — the demo knowledge base (facts the agent retrieves).
- `web_chat.py` — the Wine Sommelier web front end (Flask, localhost-only).
- `kb_wine/` — curated wine reference knowledge; `load_xwines.py` — X-Wines dataset loader.
- `make_context_diagram.py` → `Context-Diagram.pdf` — 3-page pack: system context diagram,
  a one-page write-up of what it does, and an ins/outs data-flow page.
- `finetune/` — QLoRA scaffold: behavioral training data, runner, and instructions.
- `make_howto_pdf.py` → `How-To-Use.pdf` — the printable how-to (regenerate with
  `uv run make_howto_pdf.py`).
- `sandbox/` — sample files for the calibration task (incl. hyphenated names and an
  intentionally empty file, to probe the report's exact failure modes).
- `gb10-private-llm-report-sanitized (1).html` — the original build report this mirrors.
