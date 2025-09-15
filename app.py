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
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
TROCR_SPACE_ID = os.getenv("TROCR_SPACE_ID", "akhaliq/TrOCR")  # fallback Space


# ---------------- OCR (API) ----------------
# --- OCR call with robust parsing + fallback Space ---
def extract_text(image: Image.Image) -> str:
    """
    Call HF Inference API for TrOCR with robust parsing and a fallback to a public Space.
    Raises RuntimeError with a clear message if all attempts fail.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing. In Spaces, add it under Settings → Repository secrets.")

    import io
    from huggingface_hub import InferenceClient
    from gradio_client import Client, file as gradio_file

    # Prepare image bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    notes = []

    # 1) Official client (image_to_text)
    try:
        client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=90)
        out = client.image_to_text(image=image, wait_for_model=True)
        if isinstance(out, str) and out.strip():
            return out.strip()
        if isinstance(out, list) and out and isinstance(out[0], dict) and out[0].get("generated_text"):
            return out[0]["generated_text"].strip()
        notes.append("[client] empty/unknown response")
    except Exception as e:
        notes.append(f"[client error] {e}")

    # Helper to parse JSON safely and report status/text on failure
    def _try_json(resp):
        try:
            return resp.json(), None
        except Exception:
            text = resp.text[:240].replace("\n", " ")
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

    # 3) Raw POST (multipart form)
    try:
        r = requests.post(
            API_URL + "?wait_for_model=true",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Accept": "application/json",
            },
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

    # 4) Fallback: call a public TrOCR Space via gradio_client
    try:
        c = Client(TROCR_SPACE_ID)
        result = c.predict(gradio_file(io.BytesIO(img_bytes)), api_name="/predict")
        if isinstance(result, list) and result:
            result = result[0]
        if isinstance(result, str) and result.strip():
            return result.strip()
        notes.append("[space] empty/unknown response")
    except Exception as e:
        notes.append(f"[space error] {e}")

    raise RuntimeError("OCR failed. " + " | ".join(notes))



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
