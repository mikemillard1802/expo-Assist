# Assistant_streamlit.py — cloud LLM chat + DDG search + crew research panel
import os
from typing import Optional

import streamlit as st
from openai import OpenAI

from Assistant_daily import deep_research, quick_search

st.set_page_config(
    page_title="Hybrid Assistant",
    page_icon="🦾",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; max-width: 820px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Secrets → env ────────────────────────────────────────────────────
def _secret(name: str) -> Optional[str]:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)

for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "HF_API_KEY", "HUGGINGFACE_API_KEY"):
    val = _secret(key)
    if val:
        os.environ[key] = val

# ── Cloud LLM providers (no Ollama) ──────────────────────────────────
# Order: Groq (fast) → OpenRouter → Hugging Face
PROVIDERS = []

if os.environ.get("GROQ_API_KEY"):
    PROVIDERS.append(
        {
            "name": "Groq",
            "model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": os.environ["GROQ_API_KEY"],
        }
    )

if os.environ.get("OPENROUTER_API_KEY"):
    PROVIDERS.append(
        {
            "name": "OpenRouter",
            "model": "meta-llama/llama-3.3-70b-instruct",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.environ["OPENROUTER_API_KEY"],
        }
    )

# HF OpenAI-compatible router (Inference Providers / router)
_hf_key = os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY")
if _hf_key:
    PROVIDERS.append(
        {
            "name": "Hugging Face",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "base_url": "https://router.huggingface.co/v1",
            "api_key": _hf_key,
        }
    )


def cloud_chat(messages: list, temperature: float = 0.2) -> tuple[str, str]:
    """
    Try providers in order. Returns (text, provider_name).
    Raises RuntimeError if all fail.
    """
    if not PROVIDERS:
        raise RuntimeError(
            "No cloud API keys found. Set GROQ_API_KEY, OPENROUTER_API_KEY, "
            "and/or HF_API_KEY in Streamlit secrets or environment."
        )

    errors = []
    for p in PROVIDERS:
        try:
            client = OpenAI(base_url=p["base_url"], api_key=p["api_key"])
            resp = client.chat.completions.create(
                model=p["model"],
                messages=messages,
                temperature=temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text, p["name"]
            errors.append(f"{p['name']}: empty response")
        except Exception as e:
            errors.append(f"{p['name']}: {e}")
            continue

    raise RuntimeError("All cloud providers failed:\n" + "\n".join(errors))


# ── Session state ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a concise assistant. "
                "Suggest /research <goal> for deep multi-agent jobs, "
                "or search: <query> for DuckDuckGo web lookup."
            ),
        }
    ]
if "research_output" not in st.session_state:
    st.session_state.research_output = ""
if "active_provider" not in st.session_state:
    st.session_state.active_provider = PROVIDERS[0]["name"] if PROVIDERS else "None"

# ── Header ───────────────────────────────────────────────────────────
st.title("🦾 Hybrid Assistant")
st.write("Chat uses **cloud LLMs** (Groq → OpenRouter → HF). Search uses **DuckDuckGo**.")
st.caption("Deep research runs the crew · Artifacts → `daily_artifacts/`")

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Modes")
    st.markdown("- **Chat** — cloud LLM (not Ollama)")
    st.markdown("- **`search:`** — DuckDuckGo + cloud summary")
    st.markdown("- **`/research`** — full crew")
    st.markdown("- **Research panel** — crew without slash command")
    st.markdown("---")
    st.markdown("### Active stack")
    if PROVIDERS:
        st.success("Cloud LLM · no local Ollama")
        for p in PROVIDERS:
            st.caption(f"• {p['name']}: `{p['model']}`")
        st.info(f"Last chat provider: **{st.session_state.active_provider}**")
    else:
        st.error("No API keys in secrets")
    st.markdown("---")
    st.markdown("### Tips")
    st.markdown("- Search is always external (DDG)")
    st.markdown("- Chat never calls localhost Ollama")
    st.markdown("- Crew may still use its own LLM config in `Assistant_daily`")
    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()
    if st.button("Clear research output"):
        st.session_state.research_output = ""
        st.rerun()


def _truncate(text: str, limit: int = 8000) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n\n…_(truncated)_"


def _messages_for_api():
    """OpenAI-format messages (skip empty)."""
    out = []
    for m in st.session_state.messages:
        if m.get("content"):
            out.append({"role": m["role"], "content": m["content"]})
    return out


# ── Chat panel ───────────────────────────────────────────────────────
st.subheader("💬 Chat")

for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Message, or /research <goal>, or search: <query>")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        lower = prompt.lower().strip()

        if lower.startswith("/research"):
            goal = prompt[len("/research") :].strip()
            if not goal:
                ans = "Usage: `/research <goal>`"
            else:
                with st.spinner("Crew running (may take several minutes)..."):
                    try:
                        ans = deep_research(goal, auto_confirm=True)
                        st.session_state.research_output = ans
                    except Exception as e:
                        ans = f"Crew error: {e}"
            st.markdown(_truncate(ans))

        elif lower.startswith("search:"):
            q = prompt[len("search:") :].strip()
            if not q:
                ans = "Usage: `search: <query>`"
            else:
                with st.spinner("DuckDuckGo search + cloud summary..."):
                    try:
                        snip = quick_search(q)  # external DDG
                        summary_messages = _messages_for_api() + [
                            {
                                "role": "user",
                                "content": f"Summarize briefly for the user:\n\n{snip[:3000]}",
                            }
                        ]
                        ans, prov = cloud_chat(summary_messages, temperature=0.2)
                        st.session_state.active_provider = prov
                    except Exception as e:
                        ans = f"Search error: {e}"
            st.markdown(ans)

        else:
            with st.spinner("Cloud LLM..."):
                try:
                    ans, prov = cloud_chat(_messages_for_api(), temperature=0.2)
                    st.session_state.active_provider = prov
                except Exception as e:
                    ans = f"Chat error: {e}"
            st.markdown(ans)

    st.session_state.messages.append({"role": "assistant", "content": ans})

st.divider()

# ── Research / Crew panel ────────────────────────────────────────────
st.subheader("🔬 Research / Crew")
st.write(
    "Quick search uses **DuckDuckGo** only. "
    "**Run crew** calls `deep_research` (configure that module for cloud if needed)."
)

research_goal = st.text_area(
    "Goal or idea to research / validate",
    height=120,
    placeholder="e.g. Market signals for agentic AI governance training in 2026",
    key="research_goal_box",
)

col1, col2 = st.columns(2)
with col1:
    do_search = st.button("🔍 Quick search (DDG)", use_container_width=True)
with col2:
    do_crew = st.button("🚀 Run crew", type="primary", use_container_width=True)

if do_search:
    q = (research_goal or "").strip()
    if not q:
        st.warning("Enter a query in the research box.")
    else:
        with st.spinner("DuckDuckGo..."):
            try:
                snip = quick_search(q)
                st.session_state.research_output = snip
                st.success("DDG search complete")
            except Exception as e:
                st.error(f"Search error: {e}")

if do_crew:
    goal = (research_goal or "").strip()
    if not goal:
        st.warning("Enter a goal in the research box.")
    else:
        with st.spinner("Crew running..."):
            try:
                out = deep_research(goal, auto_confirm=True)
                st.session_state.research_output = out
                st.success("Crew complete")
            except Exception as e:
                st.error(f"Crew error: {e}")

if st.session_state.research_output:
    st.markdown("### Output")
    st.markdown(_truncate(st.session_state.research_output, 12000))
    st.download_button(
        label="Download report",
        data=st.session_state.research_output,
        file_name="research_report.md",
        mime="text/markdown",
    )

st.divider()
st.caption("Cloud chat · DuckDuckGo search · Crew research panel")
