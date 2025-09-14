import os
import io
import re
import requests
from PIL import Image
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# Load HF_TOKEN from .env for local runs (ignored in HF Spaces which use Secrets)
load_dotenv()

MODEL_ID = "microsoft/trocr-base-printed"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/octet-stream"}

# -------------- OCR --------------
def extract_text(image: Image.Image) -> str:
    """
    Robust call to HF Inference API for TrOCR with detailed diagnostics.
    Tries official InferenceClient, then raw requests in two styles (octet-stream, multipart).
    Only returns after trying all paths, so we don't exit early on 404.
    """
    # Prep
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    url = API_URL + "?wait_for_model=true"

    def _dbg(label, r):
        body = r.text if hasattr(r, "text") else str(r)
        return f"[{label}] status={getattr(r,'status_code','?')} body[:240]={body[:240]!r}"

    notes = []

    # 1) Official client
    try:
        client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=90)
        out = client.image_to_text(image=image)
        if isinstance(out, str) and out.strip():
            return out.strip()
        if isinstance(out, list) and out and isinstance(out[0], dict) and out[0].get("generated_text"):
            return out[0]["generated_text"].strip()
        notes.append("[client] empty/unknown shape")
    except Exception as e:
        notes.append(f"[client error] {e}")

    # 2) Raw POST (octet-stream)
    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/octet-stream"},
            data=img_bytes,
            timeout=90,
        )
        try:
            data = r.json()
            if isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get("generated_text"):
                return data[0]["generated_text"].strip()
            if isinstance(data, dict) and isinstance(data.get("generated_text"), str):
                return data["generated_text"].strip()
            if isinstance(data, dict) and "estimated_time" in data:
                notes.append(f"[octet] model loading ~{data['estimated_time']}s")
            elif isinstance(data, dict) and "error" in data:
                notes.append(f"[octet] HF API error: {data['error']}")
            else:
                notes.append("[octet] unknown JSON shape")
        except Exception:
            notes.append(_dbg("octet-stream", r))
    except Exception as e:
        notes.append(f"[octet error] {e}")

    # 3) Raw POST (multipart form)
    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            files={"inputs": ("image.png", img_bytes, "image/png")},
            timeout=90,
        )
        try:
            data = r.json()
            if isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get("generated_text"):
                return data[0]["generated_text"].strip()
            if isinstance(data, dict) and isinstance(data.get("generated_text"), str):
                return data["generated_text"].strip()
            if isinstance(data, dict) and "estimated_time" in data:
                notes.append(f"[multipart] model loading ~{data['estimated_time']}s")
            elif isinstance(data, dict) and "error" in data:
                notes.append(f"[multipart] HF API error: {data['error']}")
            else:
                notes.append("[multipart] unknown JSON shape")
        except Exception:
            notes.append(_dbg("multipart", r))
    except Exception as e:
        notes.append(f"[multipart error] {e}")

    # If nothing worked, surface all notes for debugging in the UI
    return " | ".join(notes)




# -------------- Matching helpers --------------
import re
from difflib import get_close_matches

def _normalize(txt: str) -> str:
    # lowercase, collapse whitespace
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
    # words only (throw away punctuation); good for OCR noise like "salt."
    return re.findall(r"[a-z]+", txt.lower())


def _highlight(html_text: str, words: list) -> str:
    """Wrap allergen hits in <mark> with case-insensitive replace."""
    out = html_text
    for w in sorted(set(words), key=len, reverse=True):
        if not w:
            continue
        # Use regex to preserve original casing while highlighting
        pattern = re.compile(re.escape(w), flags=re.IGNORECASE)
        out = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", out)
    return out


# -------------- Main logic --------------
def scan_image(image, allergens_csv: str):
    if image is None:
        return gr.HTML("<b>No image uploaded.</b>")

    raw_text = extract_text(image)
    norm_text = _normalize(raw_text)

    # Show what OCR actually read (first 600 chars) so user can verify
    preview = (norm_text[:600] + ("..." if len(norm_text) > 600 else "")).replace("\n", " ")

    # Prepare allergen list
    allergens = [a.strip().lower() for a in (allergens_csv or "").split(",") if a.strip()]

    # Build token list from OCR text
    tokens = _tokenize(norm_text)
    token_set = set(tokens)

    found = []
    for a in allergens:
        hit = False

        # 1) exact token hit or variant token hit
        for v in _variants(a):
            if v in token_set:
                hit = True
                break

        # 2) fallback: regex word-boundary match in normalized text
        if not hit:
            for v in _variants(a):
                if re.search(rf"\b{re.escape(v)}\b", norm_text):
                    hit = True
                    break

        # 3) fuzzy match: catch small OCR typos
        if not hit:
            close = get_close_matches(a, tokens, n=1, cutoff=0.86)
            if close:
                hit = True

        if hit:
            found.append(a)

    # Build HTML result
    found_str = ", ".join(found) if found else "None"
    # Highlight only exact allergen terms (not all variants) in the preview text shown below
    highlighted_preview = _highlight(preview, found)

    html = f"""
    <div style="line-height:1.5">
      <h3 style="margin:0">Detected allergens: {found_str}</h3>
      <p style="margin:.25rem 0 .5rem 0;font-size:.95rem;color:#bbb">Model: <code>{MODEL_ID}</code></p>
      <h4 style="margin:.25rem 0">Extracted text (preview):</h4>
      <div style="font-family:monospace;white-space:pre-wrap">{highlighted_preview}</div>
      {"<p style='color:#f88'><b>Note:</b> " + raw_text + "</p>" if raw_text.startswith("[") else ""}
    </div>
    """
    return gr.HTML(html)


# -------------- Gradio UI --------------
with gr.Blocks() as demo:
    gr.Markdown("## 🥗 Allergen Scanner — API (Microsoft TrOCR on Hugging Face)")
    with gr.Row():
        img = gr.Image(type="pil", label="Upload ingredients photo / label")
        allergens = gr.Textbox(
            label="Your allergens (comma-separated)",
            placeholder="e.g. peanuts, milk, soy, gluten"
        )
    out = gr.HTML()
    btn = gr.Button("Scan", variant="primary")
    btn.click(scan_image, inputs=[img, allergens], outputs=[out])

if __name__ == "__main__":
    # You can set server_name="0.0.0.0" if you want LAN access
    demo.launch()
