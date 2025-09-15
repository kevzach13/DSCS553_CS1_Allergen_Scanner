import os
import io
import re
import requests
from PIL import Image
import gradio as gr
from dotenv import load_dotenv
from difflib import get_close_matches
from huggingface_hub import InferenceClient


# Load .env only for local runs. In HF Spaces, set HF_TOKEN as a Repo Secret.
load_dotenv()

# Allow overriding the model from an env var (handy in Spaces)
MODEL_ID = os.getenv("OCR_MODEL_ID", "microsoft/trocr-base-printed")
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HF_TOKEN = os.getenv("HF_TOKEN")


# ---------------- OCR (API) ----------------
def extract_text(image: Image.Image) -> str:
    """
    Robust call to HF Inference API for TrOCR with two fallbacks.
    Raises RuntimeError with a short message if all attempts fail.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    url = API_URL + "?wait_for_model=true"

    notes = []

    # 1) Official client
    try:
        client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=90)
        out = client.image_to_text(image=image)
        if isinstance(out, str) and out.strip():
            return out.strip()
        if isinstance(out, list) and out and isinstance(out[0], dict) and out[0].get("generated_text"):
            return out[0]["generated_text"].strip()
        notes.append("[client] empty/unknown response")
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
                notes.append(f"[octet] model warming (~{data['estimated_time']}s)")
            elif isinstance(data, dict) and "error" in data:
                notes.append(f"[octet] {data['error']}")
            else:
                notes.append("[octet] unknown JSON")
        except Exception as e:
            notes.append(f"[octet parse] {e}")
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
                notes.append(f"[multipart] model warming (~{data['estimated_time']}s)")
            elif isinstance(data, dict) and "error" in data:
                notes.append(f"[multipart] {data['error']}")
            else:
                notes.append("[multipart] unknown JSON")
        except Exception as e:
            notes.append(f"[multipart parse] {e}")
    except Exception as e:
        notes.append(f"[multipart error] {e}")

    raise RuntimeError("OCR failed. " + " | ".join(notes))


# ---------------- Matching helpers ----------------
def _normalize(txt: str) -> str:
    # lowercase + collapse whitespace
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
    # words only (tolerant to OCR punctuation)
    return re.findall(r"[a-z]+", txt.lower())

def _highlight(html_text: str, words: list) -> str:
    """Wrap matches in <mark> (case-insensitive)."""
    out = html_text
    for w in sorted(set(words), key=len, reverse=True):
        if not w:
            continue
        pattern = re.compile(re.escape(w), flags=re.IGNORECASE)
        out = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", out)
    return out


# ---------------- Main logic ----------------
def scan_image(image, allergens_csv: str, show_text: bool):
    # Token guard: friendlier message if secrets not set
    if not HF_TOKEN:
        return gr.HTML(
            "<b>Missing HF_TOKEN.</b> In local runs, set it in a .env file. "
            "In Hugging Face Spaces, add it under Settings → Repository secrets."
        )

    if image is None or not (allergens_csv or "").strip():
        return gr.HTML("<b>Provide an image and at least one allergen (comma-separated).</b>")

    try:
        raw_text = extract_text(image)
    except Exception as e:
        # Short, safe message (no raw JSON/error blobs)
        return gr.HTML(f"<b>OCR error:</b> {str(e)}")

    norm_text = _normalize(raw_text)
    preview = (norm_text[:600] + ("..." if len(norm_text) > 600 else "")).replace("\n", " ")

    # Prepare allergen list
    allergens = [a.strip().lower() for a in (allergens_csv or "").split(",") if a.strip()]

    tokens = _tokenize(norm_text)
    token_set = set(tokens)

    found = []
    for a in allergens:
        hit = False
        # 1) exact/variant token hit
        for v in _variants(a):
            if v in token_set:
                hit = True
                break
        # 2) regex word-boundary fallback
        if not hit:
            for v in _variants(a):
                if re.search(rf"\b{re.escape(v)}\b", norm_text):
                    hit = True
                    break
        # 3) light fuzzy (handles small OCR typos)
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
        Model: <code>{MODEL_ID}</code> • Data is processed in memory only.
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
with gr.Blocks(title="Allergen Scanner — API (TrOCR)") as demo:
    gr.Markdown("## 🥗 Allergen Scanner — API (Microsoft TrOCR on Hugging Face)")
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
    # For local testing: demo.launch(server_name="0.0.0.0", server_port=7860)
    demo.launch()
