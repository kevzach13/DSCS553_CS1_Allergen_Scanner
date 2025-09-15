import os
import io
import re
import requests
from PIL import Image
import gradio as gr
from dotenv import load_dotenv
from difflib import get_close_matches

# ---------------- Config ----------------
load_dotenv()  # local only; in Spaces use repo secrets/variables

OCRSPACE_API_KEY = os.getenv("OCRSPACE_API_KEY")  # <-- add as Repo Secret in your Space
OCRSPACE_URL = "https://api.ocr.space/parse/image"

print("OCRSPACE_API_KEY set:", bool(OCRSPACE_API_KEY))

# ---------------- OCR via OCR.space ----------------
def extract_text(image: Image.Image) -> str:
    """
    Call OCR.space REST API and return extracted text.
    Raises RuntimeError with a clear message if the call fails.
    """
    if not OCRSPACE_API_KEY:
        raise RuntimeError("OCRSPACE_API_KEY missing. In Spaces, add it under Settings → Repository secrets.")

    # Convert PIL image → bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Build request
    files = {"file": ("image.png", img_bytes, "image/png")}
    data = {
        "language": "eng",
        "scale": "true",          # helps small text
        "isTable": "false",
        "OCREngine": 2,           # 1 or 2 (2 is generally better for printed)
    }
    headers = {"apikey": OCRSPACE_API_KEY}

    try:
        r = requests.post(OCRSPACE_URL, files=files, data=data, headers=headers, timeout=90)
    except Exception as e:
        raise RuntimeError(f"OCR API request failed: {e}")

    # Robust JSON handling with diagnostics
    try:
        j = r.json()
    except Exception:
        snippet = (r.text or "")[:240].replace("\n", " ")
        raise RuntimeError(f"OCR API non-JSON response (status {r.status_code}): {snippet!r}")

    if j.get("IsErroredOnProcessing"):
        emsg = j.get("ErrorMessage") or j.get("ErrorMessageDetails") or "Unknown OCR error"
        # Some returns have list error message
        if isinstance(emsg, list) and emsg:
            emsg = emsg[0]
        raise RuntimeError(f"OCR API error: {emsg}")

    results = j.get("ParsedResults") or []
    text_parts = []
    for res in results:
        t = res.get("ParsedText") or ""
        if t.strip():
            text_parts.append(t)

    text = "\n".join(text_parts).strip()
    if not text:
        raise RuntimeError("OCR returned empty text.")

    return text

# ---------------- Matching helpers ----------------
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
        pattern = re.compile(re.escape(w), flags=re.IGNORECASE)
        out = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", out)
    return out

# ---------------- Main logic ----------------
def scan_image(image, allergens_csv: str, show_text: bool):
    if image is None or not (allergens_csv or "").strip():
        return gr.HTML("<b>Provide an image and at least one allergen (comma-separated).</b>")

    try:
        raw_text = extract_text(image)
    except Exception as e:
        return gr.HTML(f"<b>OCR error:</b> {str(e)}")

    norm_text = _normalize(raw_text)
    preview = (norm_text[:600] + ("..." if len(norm_text) > 600 else "")).replace("\n", " ")

    allergens = [a.strip().lower() for a in (allergens_csv or "").split(",") if a.strip()]
    tokens = _tokenize(norm_text)
    token_set = set(tokens)

    found = []
    for a in allergens:
        hit = False
        for v in _variants(a):
            if v in token_set:
                hit = True
                break
        if not hit:
            close = get_close_matches(a, tokens, n=1, cutoff=0.86)
            if close:
                hit = True
        if hit:
            found.append(a)

    found_str = ", ".join(found) if found else "None"
    highlighted_preview = _highlight(preview, found if show_text else [])

    html = f"""
    <div style="line-height:1.5">
      <h3 style="margin:0">Detected allergens: {found_str}</h3>
      <p style="margin:.25rem 0 .5rem 0;font-size:.95rem;color:#999">
        API: <code>OCR.space</code> • Data processed in memory only (we do not store inputs).
      </p>
      <details {"open" if show_text else ""} style="margin:.25rem 0">
        <summary style="cursor:pointer"><b>Extracted text (preview)</b></summary>
        <div style="font-family:monospace;white-space:pre-wrap;margin-top:.5rem">{highlighted_preview}</div>
      </details>
      <p style="color:#666;font-size:.9rem;margin-top:.5rem">
        Assistive tool — always verify on the original packaging. Not medical advice.
      </p>
    </div>
    """
    return gr.HTML(html)

# ---------------- Gradio UI ----------------
with gr.Blocks(title="Allergen Scanner — API (OCR.space)") as demo:
    gr.Markdown("## 🥗 Allergen Scanner — API (OCR.space)")
    with gr.Row():
        img = gr.Image(type="pil", label="Upload ingredients photo / label")
        allergens = gr.Textbox(
            label="Your allergens (comma-separated)",
            placeholder="e.g. peanuts, milk, soy, gluten"
        )
    show_text = gr.Checkbox(value=False, label="Show extracted text preview")
    out = gr.HTML()
    btn = gr.Button("Scan", variant="primary")
    btn.click(scan_image, inputs=[img, allergens, show_text], outputs=[out])

if __name__ == "__main__":
    demo.launch()


