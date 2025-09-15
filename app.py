import os
import io
import re
import requests
from PIL import Image
import gradio as gr
from dotenv import load_dotenv
from difflib import get_close_matches
from huggingface_hub import InferenceClient

# ---------------- Env & config ----------------
load_dotenv()  # for local runs only; Spaces will use repo secrets/variables

# Defaults + env
DEFAULT_MODEL_ID = "microsoft/trocr-base-printed"
MODEL_ID = (os.getenv("OCR_MODEL_ID") or DEFAULT_MODEL_ID).strip()
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

def _model_exists(mid: str) -> bool:
    try:
        # Public model metadata endpoint (NOT the inference endpoint)
        resp = requests.get(f"https://huggingface.co/api/models/{mid}", timeout=15,
                            headers={"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {})
        return resp.status_code == 200
    except Exception as e:
        print("[WARN] Model check error:", e)
        return False

# Validate MODEL_ID early and fall back if needed
if not _model_exists(MODEL_ID):
    print(f"[WARN] MODEL_ID {MODEL_ID!r} not found. Falling back to {DEFAULT_MODEL_ID!r}.")
    MODEL_ID = DEFAULT_MODEL_ID

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
print("MODEL_ID:", repr(MODEL_ID), "HF_TOKEN set:", bool(HF_TOKEN))


# ---------------- OCR (Inference API only) ----------------
def extract_text(image: Image.Image) -> str:
    """
    Call HF Inference API for TrOCR with robust parsing (client + raw HTTP).
    Raises RuntimeError with a clear message if all attempts fail.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing. In Spaces, add it under Settings → Repository secrets.")

    # prepare bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    notes = []

    # 1) Official client call
    try:
        client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=90)
        out = client.image_to_text(image=image)  # do NOT pass wait_for_model here
        if isinstance(out, str) and out.strip():
            return out.strip()
        if isinstance(out, list) and out and isinstance(out[0], dict) and out[0].get("generated_text"):
            return out[0]["generated_text"].strip()
        notes.append("[client] empty/unknown response")
    except Exception as e:
        notes.append(f"[client error] {e}")

    # helper to parse json or capture status/body
    def _try_json(resp):
        try:
            return resp.json(), None
        except Exception:
            text = (resp.text or "")[:240].replace("\n", " ")
            return None, f"status={resp.status_code} body[:240]={text!r}"

    # 2) Raw POST (octet-stream)
    try:
        r = requests.post(
            API_URL + "?wait_for_model=true",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Accept": "application/json",
                "Content-Type": "application/octet-stream",
            },
            data=img_bytes,
            timeout=90,
        )
        data, dbg = _try_json(r)
        if data:
            if isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get("generated_text"):
                return data[0]["generated_text"].strip()
            if isinstance(data, dict) and isinstance(data.get("generated_text"), str):
                return data["generated_text"].strip()
            if "estimated_time" in data:
                notes.append(f"[octet] model warming (~{data['estimated_time']}s)")
            elif "error" in data:
                notes.append(f"[octet] {data['error']}")
            else:
                notes.append("[octet] unknown JSON shape")
        else:
            notes.append(f"[octet parse] {dbg}")
    except Exception as e:
        notes.append(f"[octet error] {e}")

    # 3) Raw POST (multipart)
    try:
        r = requests.post(
            API_URL + "?wait_for_model=true",
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Accept": "application/json"},
            files={"inputs": ("image.png", img_bytes, "image/png")},
            timeout=90,
        )
        data, dbg = _try_json(r)
        if data:
            if isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get("generated_text"):
                return data[0]["generated_text"].strip()
            if isinstance(data, dict) and isinstance(data.get("generated_text"), str):
                return data["generated_text"].strip()
            if "estimated_time" in data:
                notes.append(f"[multipart] model warming (~{data['estimated_time']}s)")
            elif "error" in data:
                notes.append(f"[multipart] {data['error']}")
            else:
                notes.append("[multipart] unknown JSON shape")
        else:
            notes.append(f"[multipart parse] {dbg}")
    except Exception as e:
        notes.append(f"[multipart error] {e}")

    raise RuntimeError("OCR failed. " + " | ".join(notes))

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
    demo.launch()

