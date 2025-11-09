import os
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

import io 
import re
import time
import requests
from PIL import Image
import gradio as gr
from difflib import get_close_matches
from dotenv import load_dotenv

# === NEW: Prometheus app-level metrics ===
from prometheus_client import start_http_server, Counter, Histogram
start_http_server(8000)  # expose app metrics at /metrics on port 8000
APP_REQUESTS = Counter("app_requests_total", "Total app requests")
APP_LATENCY  = Histogram("app_request_latency_seconds", "Request latency (s)")

# Config 
load_dotenv()  # local only; in Docker set env vars at runtime
OCRSPACE_API_KEY = os.getenv("OCRSPACE_API_KEY")
OCRSPACE_URL = "https://api.ocr.space/parse/image"

# OCR
def extract_text(image: Image.Image) -> str:
    if not OCRSPACE_API_KEY:
        raise RuntimeError("Missing OCRSPACE_API_KEY (set it as an environment variable)")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    files = {"file": ("image.png", buf.getvalue(), "image/png")}
    data = {"language": "eng", "scale": "true", "isTable": "false", "OCREngine": 2}
    headers = {"apikey": OCRSPACE_API_KEY}

    r = requests.post(OCRSPACE_URL, files=files, data=data, headers=headers, timeout=60)
    j = r.json()  # minimal happy-path

    results = j.get("ParsedResults") or []
    text = (results[0].get("ParsedText") if results else "") or ""
    text = text.strip()
    if not text:
        raise RuntimeError("No text detected.")
    return text

# Matching helpers
def _normalize(txt: str) -> str:
    return re.sub(r"\s+", " ", (txt or "")).strip().lower()

def _variants(term: str):
    t = (term or "").strip().lower()
    c = {t}
    if t.endswith("es"): c.add(t[:-2])
    if t.endswith("s"):  c.add(t[:-1])
    c.add(t.replace("-", " "))
    c.add(t.replace(" ", ""))
    return list(c)

def _tokenize(txt: str):
    return re.findall(r"[a-z]+", txt.lower())

def _highlight(html_text: str, words: list) -> str:
    out = html_text
    for w in sorted(set(words), key=len, reverse=True):
        if not w: continue
        out = re.compile(re.escape(w), flags=re.IGNORECASE).sub(
            lambda m: f"<mark>{m.group(0)}</mark>", out
        )
    return out

# Main
def scan_image(image, allergens_csv: str, show_text: bool):
    if image is None or not (allergens_csv or "").strip():
        return gr.HTML("<div class='card warn'>Upload an image and enter allergens (comma-separated).</div>")

    # === NEW: metrics timing + counter ===
    t0 = time.perf_counter()
    APP_REQUESTS.inc()

    t_ocr0 = time.perf_counter()
    try:
        raw_text = extract_text(image)
    except Exception as e:
        return gr.HTML(f"<div class='card error'><b>OCR error:</b> {e}</div>")
    t_ocr_ms = (time.perf_counter() - t_ocr0) * 1000.0

    # keep OCR timing in logs only (not shown to user)
    print(f"[metrics] OCR_time_ms={t_ocr_ms:.1f}")

    norm_text = _normalize(raw_text)
    preview = (norm_text[:800] + ("..." if len(norm_text) > 800 else "")).replace("\n", " ")

    allergens = [a.strip().lower() for a in (allergens_csv or "").split(",") if a.strip()]
    tokens = _tokenize(norm_text)
    token_set = set(tokens)

    found = []
    for a in allergens:
        hit = any(v in token_set for v in _variants(a))
        if not hit:  # small OCR typos
            hit = bool(get_close_matches(a, tokens, n=1, cutoff=0.86))
        if hit:
            found.append(a)

    total_ms = (time.perf_counter() - t0) * 1000.0

    # === NEW: observe latency ===
    APP_LATENCY.observe(total_ms / 1000.0)

    chips = (
        "".join(f"<span class='chip hit'>{a}</span>" for a in sorted(set(found)))
        if found else "<span class='chip ok'>None</span>"
    )
    highlighted_preview = _highlight(preview, found if show_text else [])

    html = f"""
    <style>
      .wrap {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; line-height:1.45; }}
      .subtle {{ color:#6b7280; font-size:.92rem; }}
      .row {{ display:flex; gap:.75rem; flex-wrap:wrap; align-items:center; }}
      .card {{ background:#0b1220; border:1px solid #1f2a44; padding:12px 14px; border-radius:12px; }}
      .warn {{ border-color:#665200; background:#1f1a00; }}
      .error {{ border-color:#7f1d1d; background:#1f0b0b; }}
      .chip {{ display:inline-block; padding:.25rem .6rem; border-radius:999px; font-weight:600; }}
      .chip.hit {{ background:#fee2e2; color:#991b1b; border:1px solid #fecaca; }}
      .chip.ok  {{ background:#dcfce7; color:#166534; border:1px solid #bbf7d0; }}
      .metric {{ display:inline-block; background:#0b1220; border:1px solid #1f2a44; border-radius:8px; padding:.25rem .5rem; }}
      details {{ margin-top:.5rem; }} summary {{ cursor:pointer; }}
    </style>
    <div class="wrap">
      <div class="card">
        <div class="row" style="justify-content:space-between">
          <div>
            <div style="font-size:1.1rem;font-weight:700;">Detected allergens</div>
            <div class="row" style="margin-top:.35rem">{chips}</div>
          </div>
          <div class="row" style="justify-content:flex-end;text-align:right">
            <span class="metric">Time: {total_ms:.1f} ms</span>
          </div>
        </div>
        <details {"open" if show_text else ""}>
          <summary><b>Extracted text (preview)</b></summary>
          <div style="font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space:pre-wrap; margin-top:.5rem;">
            {highlighted_preview}
          </div>
        </details>
        <div class="subtle" style="margin-top:.5rem">
          Assistive tool — always verify on the original packaging. Not medical advice.
        </div>
      </div>
    </div>
    """
    return gr.HTML(html)

# UI
with gr.Blocks(title="Allergen Scanner — API (OCR.space)") as demo:
    gr.Markdown("## 🥗 Allergen Scanner — API (OCR.space)")
    with gr.Row():
        img = gr.Image(type="pil", label="Upload ingredients photo / label")
        allergens = gr.Textbox(label="Your allergens (comma-separated)",
                               placeholder="e.g. peanuts, milk, soy, gluten")
    show_text = gr.Checkbox(value=False, label="Show extracted text preview")
    out = gr.HTML()
    gr.Button("Scan", variant="primary").click(scan_image, [img, allergens, show_text], [out])

if __name__ == "__main__":
    # === IMPORTANT: bind to 0.0.0.0 and port 7860 for Docker ===
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

