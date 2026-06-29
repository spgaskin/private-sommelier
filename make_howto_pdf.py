"""
make_howto_pdf.py - generate How-To-Use.pdf for the private LLM stack.

Pure-Python (fpdf2), no system tools. Run: uv run make_howto_pdf.py
Text is ASCII so it renders with the built-in core fonts.
"""

from fpdf import FPDF

BLUE = (39, 96, 146)
INK = (15, 27, 42)
MUTED = (86, 102, 117)
CODE_BG = (233, 240, 246)
GUARD = (158, 59, 46)

# (kind, text) blocks. kinds: h1, h2, p, bullet, code, note, guard, rule, space
DOC = [
    ("h1", "Private LLM - How To Use"),
    ("p", "A self-contained private LLM + agent on your Mac (Apple M5, 32 GB). An open "
          "model runs locally, served over a localhost-only API, driving a tool-calling "
          "agent. Nothing leaves the machine by default. This mirrors the GB10 build "
          "report, scaled to Apple Silicon."),
    ("h2", "1. The architecture"),
    ("code", "open model (Qwen3 8B)\n   -> Ollama OpenAI endpoint @ 127.0.0.1:11434\n"
             "      -> agent loop: goal -> pick tool -> act -> report"),
    ("p", "Facts live in a retrieval layer (rag.py + kb/), not baked into the model. "
          "Behavior can be shaped by fine-tuning (finetune/). This is the report's "
          "'behavior in weights, facts in retrieval' split."),
    ("h2", "2. One-time setup (already done in this folder)"),
    ("bullet", "Ollama installed via Homebrew, running as a localhost service "
               "(brew services start ollama)."),
    ("bullet", "Models pulled into ~/.ollama: qwen3:8b, granite3.3:8b, nomic-embed-text."),
    ("bullet", "Python env managed by uv (Python 3.11). No global installs."),
    ("note", "If the server is ever down: brew services start ollama   "
             "(check: curl http://127.0.0.1:11434/api/version )"),
    ("h2", "3. Run the agent"),
    ("p", "All commands run from inside the 'Private LLM' folder."),
    ("code", "# the report's calibration task: inventory a folder\n"
             "uv run private_agent.py --task inventory --dir ./sandbox\n\n"
             "# a factual question (answered from the knowledge base)\n"
             "uv run private_agent.py --goal \"What port is billing on? Use retrieve.\"\n\n"
             "# let the agent write a file (opt-in, sandboxed)\n"
             "uv run private_agent.py --allow-write \\\n"
             "    --goal \"Read people.csv and write a summary to summary.txt.\""),
    ("h2", "4. The tools the agent has"),
    ("bullet", "list_dir, read_file, file_info - always on, read-only."),
    ("bullet", "retrieve - always on, searches the local knowledge base (kb/)."),
    ("bullet", "write_file - opt-in with --allow-write, writes inside the sandbox only."),
    ("guard", "fetch_url - opt-in with --allow-web. THIS LEAVES THE BOX and breaks the "
              "sovereignty story. Use only when you intend network access."),
    ("h2", "5. The knowledge base (RAG)"),
    ("p", "Facts go in kb/ as plain text/markdown. Re-index after editing:"),
    ("code", "uv run rag.py ingest --dir ./kb          # rebuild the index\n"
             "uv run rag.py query \"who is the architect?\"   # search it directly"),
    ("p", "The agent's retrieve tool searches this same index, so updating a fact is just "
          "editing a file in kb/ and re-ingesting - no retraining."),
    ("h2", "6. Fine-tuning (the next phase)"),
    ("p", "Tune the model's behavior (cite sources, refuse to guess) while keeping facts in "
          "RAG. QLoRA via MLX on Apple Silicon:"),
    ("code", "cd finetune\n./run_finetune.sh        # writes LoRA adapters to ./adapters"),
    ("p", "See finetune/README.md to fuse the adapters and serve the tuned model through "
          "the same local endpoint."),
    ("h2", "7. Swapping the model"),
    ("code", "PRIVATE_LLM_MODEL=granite3.3:8b uv run private_agent.py --task inventory"),
    ("p", "Default is qwen3:8b because it drives the tool loop reliably. granite3.3:8b is "
          "faithful to the report but flaky through Ollama (see troubleshooting)."),
    ("h2", "8. Troubleshooting / findings"),
    ("bullet", "Agent hallucinates instead of calling tools: this is the report's core "
               "finding for small (8B) models. Prefer qwen3:8b; keep guidance in the user "
               "turn, not a heavy system prompt (that suppresses tool calls)."),
    ("bullet", "Model invents file sizes/contents: it skipped a tool. Trust only what the "
               "tool trace shows; verify anything an 8B asserts unsupervised."),
    ("bullet", "Connection refused: the Ollama service isn't running - start it (section 2)."),
    ("h2", "9. Sovereignty posture"),
    ("p", "By default everything is localhost-bound and read-only: the model endpoint and "
          "the agent never touch the network. Writes are opt-in and sandboxed; web access "
          "is opt-in and clearly flagged as leaving the box. That is governance by "
          "construction - the same principle as the report."),
]


class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 10, f"Private LLM - How To Use   |   page {self.page_no()}", align="C")


def render():
    pdf = PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)

    for kind, text in DOC:
        if kind == "h1":
            pdf.set_font("Helvetica", "B", 22)
            pdf.set_text_color(*INK)
            pdf.multi_cell(0, 9, text)
            pdf.ln(2)
            pdf.set_draw_color(*BLUE)
            pdf.set_line_width(0.6)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(4)
        elif kind == "h2":
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(*BLUE)
            pdf.multi_cell(0, 7, text)
            pdf.ln(1)
        elif kind == "p":
            pdf.set_font("Helvetica", "", 10.5)
            pdf.set_text_color(*INK)
            pdf.multi_cell(0, 5.6, text)
            pdf.ln(1.5)
        elif kind == "bullet":
            pdf.set_font("Helvetica", "", 10.5)
            pdf.set_text_color(*INK)
            x = pdf.get_x()
            pdf.set_text_color(*BLUE)
            pdf.cell(5, 5.6, chr(149))
            pdf.set_text_color(*INK)
            pdf.multi_cell(0, 5.6, text)
            pdf.set_x(x)
            pdf.ln(0.5)
        elif kind == "code":
            pdf.ln(1)
            pdf.set_font("Courier", "", 9)
            pdf.set_fill_color(*CODE_BG)
            pdf.set_text_color(*INK)
            pdf.multi_cell(0, 5, text, fill=True, border=0)
            pdf.ln(2)
        elif kind in ("note", "guard"):
            pdf.ln(1)
            color = MUTED if kind == "note" else GUARD
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.set_text_color(*color)
            label = "NOTE  " if kind == "note" else "CAUTION  "
            pdf.multi_cell(0, 5.4, label + text)
            pdf.ln(1.5)

    out = "How-To-Use.pdf"
    pdf.output(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    render()
