import os
import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

# ---------------- Helpers ----------------
def read_url_content(url: str) -> str | None:
    """Fetch URL and return visible text."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; HW2Bot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        clean = " ".join(text.split())
        return clean if clean.strip() else None
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


def build_instruction(summary_type: str) -> str:
    if summary_type == "100 words":
        return "Summarize the document in about 100 words."
    elif summary_type == "2 paragraphs":
        return "Summarize the document in exactly 2 connecting paragraphs."
    else:
        return "Summarize the document in 5 concise bullet points."


def build_prompt(document: str, instruction: str, language: str) -> str:
    return (
        f"Here’s a document: {document}\n\n---\n\n"
        f"{instruction} Write the summary strictly in {language}."
    )


def get_secret(name: str) -> str | None:
    return st.secrets.get(name, os.getenv(name))


# ---------------- Vendor / Model options ----------------
VENDORS = ["OpenAI", "Anthropic (Claude)", "Cohere", "Mistral"]

MODEL_OPTIONS = {
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-5-chat-latest"],
        "key_name": "OPENAI_API_KEY",
    },
    "Anthropic (Claude)": {
        "models": [
            "claude-3-5-haiku-20241022",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
        ],
        "key_name": "ANTHROPIC_API_KEY",
    },
    "Cohere": {
        "models": ["command-r", "command-r-plus", "command"],
        "key_name": "COHERE_API_KEY",
    },
    "Mistral": {
        "models": ["open-mistral-7b", "open-mixtral-8x7b"],
        "key_name": "MISTRAL_API_KEY",
    },
}


# ---------------- Adapter ----------------
def summarize_with(vendor: str, model: str, prompt: str):
    key_name = MODEL_OPTIONS[vendor]["key_name"]
    api_key = get_secret(key_name)

    if not api_key:
        raise RuntimeError(
            f"Missing API key for {vendor}. Set {key_name} in Secrets (or env) and restart."
        )

    if vendor == "OpenAI":
        return None, model  # caller streams directly

    if vendor == "Anthropic (Claude)":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model,
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
            return text or "", model
        except ModuleNotFoundError:
            raise RuntimeError("anthropic SDK not installed. Add `anthropic` to requirements.txt.")
        except Exception as e:
            raise RuntimeError(f"Anthropic error: {e}")

    if vendor == "Cohere":
        try:
            import cohere
            co = cohere.Client(api_key)
            resp = co.chat(model=model, message=prompt)
            text = getattr(resp, "text", "") or getattr(resp, "output_text", "")
            return text or "", model
        except ModuleNotFoundError:
            raise RuntimeError("cohere SDK not installed. Add `cohere` to requirements.txt.")
        except Exception as e:
            raise RuntimeError(f"Cohere error: {e}")

    if vendor == "Mistral":
        try:
            from mistralai import Mistral
            client = Mistral(api_key=api_key)
            resp = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content if resp and resp.choices else ""
            return text or "", model
        except ModuleNotFoundError:
            raise RuntimeError("mistralai SDK not installed. Add `mistralai>=1.0.0` to requirements.txt.")
        except Exception as e:
            raise RuntimeError(f"Mistral error: {e}")

    raise RuntimeError(f"Unsupported vendor: {vendor}")


# ---------------- UI ----------------
st.title("📄 Gautam's Document Bot — HW2 (URL Summarizer)")
st.write("Enter a URL, choose summary style, language, vendor & model, then generate your summary.")

st.sidebar.header("Summary Options")
summary_type = st.sidebar.radio(
    "Choose summary style:",
    ("100 words", "2 paragraphs", "5 bullet points"),
)

st.sidebar.header("Output Language")
language = st.sidebar.selectbox(
    "Choose language:",
    ["English", "Spanish", "French", "German", "Hindi", "Marathi"],
    index=0,
)

st.sidebar.header("Model Vendor")
vendor = st.sidebar.selectbox("Choose vendor:", options=VENDORS, index=0)

# Model dropdown
model = st.sidebar.selectbox("Choose model:", options=MODEL_OPTIONS[vendor]["models"], index=0)

needed_key = MODEL_OPTIONS[vendor]["key_name"]
has_key = bool(get_secret(needed_key))
st.sidebar.caption(
    f"Using model: **{model}**  •  Needs **{needed_key}** "
    f"{'✅' if has_key else '❌ (add in Secrets & restart)'}"
)

# Main input
url = st.text_input("Enter a URL (https://...)")
summarize = st.button("Summarize", disabled=not url.startswith(("http://", "https://")))

if summarize:
    with st.spinner("Fetching page content..."):
        document = read_url_content(url)

    if not document:
        st.error("Could not read text from that URL. It might be empty, blocked, or paywalled. Try another page.")
        st.stop()

    instruction = build_instruction(summary_type)
    prompt = build_prompt(document, instruction, language)

    if vendor == "OpenAI":
        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            st.error("Missing OPENAI_API_KEY. Add it in Secrets and restart.")
            st.stop()

        client = OpenAI(api_key=api_key)
        with st.spinner(f"Generating summary using {model} (OpenAI)…"):
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
        st.write_stream(stream)
    else:
        try:
            with st.spinner(f"Generating summary using {model} ({vendor})…"):
                output, model_used = summarize_with(vendor, model, prompt)
            if not output.strip():
                st.warning(
                    f"The model ({model_used}) returned an empty response. "
                    "Try another URL or pick a different model."
                )
            else:
                st.write(output)
        except RuntimeError as e:
            st.error(str(e))
