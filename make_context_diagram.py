"""
make_context_diagram.py - generate Context-Diagram.pdf for the private LLM stack.

Three pages, drawn with fpdf2:
  1. System context diagram (components + boundary + the two flows).
  2. A one-page write-up: what the system does.
  3. Ins & outs: the request flow, the build flow, and an inputs/outputs summary.

Run: uv run make_context_diagram.py
"""

import math

from fpdf import FPDF

# palette
INK = (24, 26, 29)
MUTED = (122, 106, 108)
WINE = (107, 31, 42)
WINE_BG = (243, 231, 227)
BLUE = (39, 96, 146)
BLUE_BG = (233, 240, 246)
GREEN = (31, 122, 92)
GREEN_BG = (227, 241, 234)
AMBER = (143, 100, 18)
AMBER_BG = (244, 236, 216)
GREY = (120, 120, 128)
GREY_BG = (238, 238, 240)
WHITE = (255, 255, 255)


class Doc(FPDF):
    # ---- primitives ----
    def box(self, x, y, w, h, title, lines=None, accent=INK, fill=WHITE, dashed=False,
            title_size=8.5, text_size=6.8):
        self.set_draw_color(*accent)
        self.set_line_width(0.5 if not dashed else 0.4)
        if dashed:
            self.set_dash_pattern(dash=1.4, gap=1.4)
        self.set_fill_color(*fill)
        self.rect(x, y, w, h, style="DF")
        if dashed:
            self.set_dash_pattern()
        self.set_fill_color(*accent)
        self.rect(x, y, 1.6, h, style="F")
        self.set_xy(x + 3.5, y + 2.6)
        self.set_font("Helvetica", "B", title_size)
        self.set_text_color(*INK)
        self.multi_cell(w - 5, 4, title, align="L")
        if lines:
            self.set_font("Helvetica", "", text_size)
            self.set_text_color(*MUTED)
            for ln in lines:
                self.set_x(x + 3.5)
                self.multi_cell(w - 5, 3.3, ln, align="L")

    def pill(self, x, y, w, h, text, accent=BLUE):
        self.set_draw_color(*accent)
        self.set_line_width(0.3)
        self.set_fill_color(*WHITE)
        self.rect(x, y, w, h, style="DF")
        self.set_xy(x, y + (h - 3) / 2)
        self.set_font("Helvetica", "B", 6.4)
        self.set_text_color(*accent)
        self.cell(w, 3, text, align="C")

    def arrow(self, x1, y1, x2, y2, label=None, color=INK, dashed=False, lbldy=-2.4, lw=0.4):
        self.set_draw_color(*color)
        self.set_line_width(lw)
        if dashed:
            self.set_dash_pattern(dash=1.3, gap=1.3)
        self.line(x1, y1, x2, y2)
        if dashed:
            self.set_dash_pattern()
        ang = math.atan2(y2 - y1, x2 - x1)
        L, spread = 2.6, math.radians(26)
        for s in (-1, 1):
            self.line(x2, y2, x2 - L * math.cos(ang + s * spread), y2 - L * math.sin(ang + s * spread))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            self.set_font("Helvetica", "", 6.2)
            self.set_text_color(*color)
            tw = self.get_string_width(label) + 2
            self.set_xy(mx - tw / 2, my + lbldy)
            self.set_fill_color(*WHITE)
            self.cell(tw, 3, label, align="C", fill=True)

    def heading(self, x, y, text, sub=None):
        self.set_xy(x, y)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*INK)
        self.cell(0, 7, text)
        if sub:
            self.set_xy(x, y + 8)
            self.set_font("Helvetica", "", 8.5)
            self.set_text_color(*MUTED)
            self.multi_cell(265, 4, sub)

    def legend(self, x, y):
        self.set_draw_color(*INK)
        self.set_line_width(0.4)
        self.line(x, y, x + 10, y)
        self.set_xy(x + 12, y - 1.6)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*MUTED)
        self.cell(45, 3, "live chat path")
        self.set_dash_pattern(dash=1.3, gap=1.3)
        self.line(x + 60, y, x + 70, y)
        self.set_dash_pattern()
        self.set_xy(x + 72, y - 1.6)
        self.cell(45, 3, "one-time data build")

    def para(self, x, w, y, kind, text):
        if kind == "h":
            self.set_xy(x, y); self.set_font("Helvetica", "B", 10.5); self.set_text_color(*WINE)
            self.multi_cell(w, 5, text); return self.get_y() + 1.4
        if kind == "p":
            self.set_xy(x, y); self.set_font("Helvetica", "", 8.8); self.set_text_color(*INK)
            self.multi_cell(w, 4.5, text); return self.get_y() + 2.6
        if kind == "b":
            self.set_xy(x, y); self.set_font("Helvetica", "", 8.6); self.set_text_color(*BLUE)
            self.cell(3.5, 4.4, chr(149))
            self.set_xy(x + 4, y); self.set_text_color(*INK)
            self.multi_cell(w - 4, 4.4, text); return self.get_y() + 1.3

    def flowbox(self, x, y, w, h, num, title, sub, accent):
        self.box(x, y, w, h, title, [sub] if sub else None, accent=accent, fill=WHITE,
                 title_size=7.6, text_size=6.4)
        # number badge
        self.set_fill_color(*accent)
        self.ellipse(x + w - 9, y - 3.5, 7, 7, style="F")
        self.set_xy(x + w - 9, y - 2.7)
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*WHITE)
        self.cell(7, 3, str(num), align="C")


def page_diagram(d):
    d.add_page(orientation="L")
    d.heading(14, 10, "Private LLM - System Context Diagram",
              "A private wine assistant on Apple Silicon. Every box runs locally and binds to 127.0.0.1; "
              "Ollama is a shared local service. The X-Wines dataset and model weights are fetched once, "
              "then it all runs offline.")

    # boundary
    d.set_draw_color(*WINE); d.set_line_width(0.6)
    d.set_dash_pattern(dash=2, gap=2); d.rect(58, 30, 226, 168); d.set_dash_pattern()
    d.set_xy(61, 31.5); d.set_font("Helvetica", "B", 7); d.set_text_color(*WINE)
    d.cell(0, 4, "YOUR MAC  .  Apple M5  .  32 GB unified  .  localhost only - nothing leaves the box")

    # shared service: Ollama (top-center)
    d.box(104, 36, 96, 38, "Ollama  -  shared local serving layer",
          ["OpenAI-compatible API  .  127.0.0.1:11434  .  runs on Metal"], accent=BLUE, fill=BLUE_BG)
    d.pill(108, 54, 44, 9, "qwen3:8b  -  chat", accent=BLUE)
    d.pill(154, 54, 42, 9, "nomic-embed-text", accent=BLUE)

    # actors / sources (outside, left)
    d.box(12, 100, 42, 28, "You", ["Web browser", "(the operator)"], accent=INK, fill=GREY_BG)
    d.box(12, 150, 42, 22, "kb_wine/", ["Curated reference notes"], accent=GREEN, fill=GREEN_BG)
    d.box(12, 176, 42, 20, "X-Wines dataset", ["100K wines . 21M ratings", "open license, fetched once"],
          accent=GREEN, fill=GREEN_BG, dashed=True)

    # inside boundary
    d.box(68, 98, 52, 34, "web_chat.py", ["Flask web app  .  :8080", "Private Sommelier UI",
                                          "streams stages over SSE"], accent=WINE, fill=WINE_BG)
    d.box(140, 100, 40, 28, "rag.py", ["Embed query +", "cosine search"], accent=GREEN, fill=GREEN_BG)
    d.box(212, 96, 62, 40, "wine_index.npz", ["The RAG index . 100,666 docs", "embeddings + source text",
                                              "facts live here, not in weights"], accent=GREEN, fill=GREEN_BG)
    d.box(140, 150, 56, 30, "load_xwines.py", ["Build pipeline", "stream ratings -> docs", "chunk -> embed -> write"],
          accent=GREY, fill=GREY_BG)
    d.box(212, 148, 62, 22, "private_agent.py", ["Tool-calling agent", "read/write/retrieve/web"],
          accent=AMBER, fill=AMBER_BG)
    d.box(212, 174, 62, 20, "finetune/", ["MLX QLoRA scaffold", "tune behaviour (next phase)"],
          accent=GREY, fill=GREY_BG, dashed=True)

    # live chat arrows (solid)
    d.arrow(54, 112, 68, 112, "POST /chat", color=WINE)
    d.arrow(120, 113, 140, 113, "retrieve", color=INK)
    d.arrow(180, 113, 212, 113, "top-5 context", color=GREEN)
    d.arrow(160, 100, 150, 74, "embed query", color=BLUE, lbldy=-2.4)
    d.arrow(96, 98, 120, 74, "chat (stream)", color=WINE, lbldy=-2.4)
    d.arrow(212, 158, 180, 120, "retrieve / tools", color=AMBER, lbldy=-2.4)

    # build arrows (dashed)
    d.arrow(54, 160, 140, 162, "curated notes", color=GREEN, dashed=True, lbldy=-2.4)
    d.arrow(54, 184, 140, 170, "load wines", color=GREEN, dashed=True, lbldy=-2.4)
    d.arrow(190, 150, 183, 74, "embed docs", color=BLUE, dashed=True, lbldy=-24)
    d.arrow(196, 168, 212, 128, "write index", color=GREY, dashed=True, lbldy=2.6)

    d.legend(150, 192)


def page_writeup(d):
    d.add_page(orientation="L")
    d.heading(14, 10, "What This System Does",
              "A plain-language tour of the private wine assistant and why it is built the way it is.")
    LX, LW, RX, RW, top = 14, 130, 154, 130, 32
    y = top
    y = d.para(LX, LW, y, "h", "Overview")
    y = d.para(LX, LW, y, "p", "A private wine sommelier that runs entirely on your Mac. You ask questions "
               "in a browser; a local language model answers, grounded in a wine knowledge base of 100,666 "
               "entries. No cloud, no API key, and no data leaves the machine.")
    y = d.para(LX, LW, y, "h", "Sovereignty by construction")
    y = d.para(LX, LW, y, "p", "Every component binds to 127.0.0.1. The model runs on Apple Silicon (Metal) "
               "through Ollama; retrieval and the web app are local processes. The only outbound step is a "
               "one-time download of the open X-Wines dataset and the model weights - after that it runs "
               "fully offline.")
    y = d.para(LX, LW, y, "h", "Two data layers")
    y = d.para(LX, LW, y, "p", "Curated knowledge (kb_wine/) explains how wine works: regions, grapes, "
               "classifications, pairing. The X-Wines dataset adds 100K real bottles with average ratings "
               "drawn from 21M reviews. Both are embedded into one index, so a single question can pull on "
               "reference knowledge and specific recommendations at once.")
    y = d.para(LX, LW, y, "h", "Facts in retrieval, not weights")
    y = d.para(LX, LW, y, "p", "The model supplies language and reasoning; the facts come from the index at "
               "query time. Update a fact by editing a file and re-indexing - no retraining, and every answer "
               "stays auditable against its sources.")

    y = top
    y = d.para(RX, RW, y, "h", "The components")
    for item in [
        "Ollama - serves qwen3:8b (chat) and nomic-embed-text (embeddings) on :11434.",
        "rag.py - embeds the question and does cosine search over the index.",
        "wine_index.npz - the embedded knowledge base (100,666 documents).",
        "web_chat.py - Flask web UI on :8080; streams its progress over SSE.",
        "load_xwines.py - builds the index from kb_wine/ and the X-Wines dataset.",
        "private_agent.py - a sandboxed tool-calling agent (read/write/retrieve/web).",
        "finetune/ - an MLX QLoRA scaffold to tune behaviour (the next phase).",
    ]:
        y = d.para(RX, RW, y, "b", item)
    y = d.para(RX, RW, y + 1, "h", "A question, end to end")
    y = d.para(RX, RW, y, "p", "Your question is embedded, matched against the index, and the top wines become "
               "context for the model, which streams an answer back to the browser with its sources. You watch "
               "each stage live: searching the cellar, the wines it found, then the answer being written.")
    y = d.para(RX, RW, y, "h", "What's next")
    y = d.para(RX, RW, y, "p", "Swap in a larger local model (qwen3:14b or 30b-a3b), run the fine-tune, or load "
               "your own cellar and tasting notes into kb_wine/ to personalise the recommendations.")


def page_flows(d):
    d.add_page(orientation="L")
    d.heading(14, 10, "Ins & Outs - How Data Flows",
              "Two paths move through the system: answering a question (live) and building the knowledge (one-time).")

    # Flow A
    d.set_xy(14, 28); d.set_font("Helvetica", "B", 10); d.set_text_color(*WINE)
    d.cell(0, 5, "Flow A  -  Answering a question  (live, every request)")
    ax, ay, aw, ah, pitch = 8, 40, 32, 24, 41
    steps_a = [
        ("Browser", "you type a question", INK),
        ("web_chat.py", "POST /chat", WINE),
        ("rag.py", "embed the query", GREEN),
        ("nomic-embed", "query -> vector", BLUE),
        ("wine_index", "cosine search -> top 5", GREEN),
        ("qwen3:8b", "stream the answer", BLUE),
        ("Browser", "renders live (SSE)", WINE),
    ]
    for i, (t, s, c) in enumerate(steps_a):
        x = ax + i * pitch
        d.flowbox(x, ay, aw, ah, i + 1, t, s, c)
        if i:
            d.arrow(ax + (i - 1) * pitch + aw, ay + ah / 2, x, ay + ah / 2, color=MUTED)

    # Flow B
    d.set_xy(14, 86); d.set_font("Helvetica", "B", 10); d.set_text_color(*GREEN)
    d.cell(0, 5, "Flow B  -  Building the index  (one-time, re-run when data changes)")
    bx, by, bw, bh, bpitch = 10, 98, 48, 26, 56
    steps_b = [
        ("Sources", "kb_wine/ + X-Wines", GREEN),
        ("Aggregate", "stream 21M ratings -> avg", GREY),
        ("Build docs", "1 per wine + curated", GREY),
        ("Embed", "batches via nomic", BLUE),
        ("wine_index.npz", "write 100,666 docs", GREEN),
    ]
    for i, (t, s, c) in enumerate(steps_b):
        x = bx + i * bpitch
        d.flowbox(x, by, bw, bh, i + 1, t, s, c)
        if i:
            d.arrow(bx + (i - 1) * bpitch + bw, by + bh / 2, x, by + bh / 2, color=MUTED, dashed=True)

    # Inputs / Outputs band
    iy = 146
    d.box(14, iy, 128, 44, "INPUTS  (what goes in)", [
        "Your questions (typed in the browser).",
        "kb_wine/ - curated reference notes you write.",
        "X-Wines dataset - 100K wines / 21M ratings, fetched once.",
        "Model weights - qwen3:8b + nomic-embed-text, pulled once.",
    ], accent=BLUE, fill=BLUE_BG, title_size=9, text_size=8)
    d.box(154, iy, 128, 44, "OUTPUTS  (what comes out)", [
        "Streamed answers with their sources (in the browser).",
        "wine_index.npz - the reusable embedded knowledge base.",
        "Agent: files written into its sandbox (when enabled).",
        "Fine-tune: LoRA adapters under finetune/ (next phase).",
    ], accent=GREEN, fill=GREEN_BG, title_size=9, text_size=8)


def render():
    d = Doc(orientation="L", format="A4")
    d.set_auto_page_break(False)
    page_diagram(d)
    page_writeup(d)
    page_flows(d)
    d.output("Context-Diagram.pdf")
    print("wrote Context-Diagram.pdf (3 pages)")


if __name__ == "__main__":
    render()
