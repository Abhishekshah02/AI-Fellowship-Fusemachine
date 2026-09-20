"""Streamlit chat UI. Talks to the FastAPI backend over HTTP -- it holds no
model, no key and no index, so the API can scale independently of it."""
from __future__ import annotations

import os

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="ShopAssist AI", page_icon="🛍️", layout="centered")
st.title("🛍️ ShopAssist AI")
st.caption("RAG + tool-calling support assistant. Answers are grounded in the policy knowledge base.")

mode = st.radio(
    "Mode",
    ["Chat (W15 · single pass)", "Investigate (W16 · agentic loop)"],
    horizontal=True,
    label_visibility="collapsed",
)
AGENT_MODE = mode.startswith("Investigate")

with st.sidebar:
    st.subheader("Backend")
    try:
        health = httpx.get(f"{API}/health", timeout=5).json()
        st.success("connected")
        st.json(health)
    except Exception as exc:
        st.error(f"API unreachable at {API}\n\n{exc}")
    if st.button("Re-index knowledge base"):
        st.write(httpx.post(f"{API}/ingest", timeout=120).json())
    if st.button("Show metrics"):
        st.json(httpx.get(f"{API}/metrics", timeout=5).json())
    st.divider()
    st.caption("**Chat:** *Where is order SA-10231?* · *Can I cancel SA-10244?*")
    st.caption("**Investigate:** *I was charged twice for SA-10231* · "
               "*My refund for SA-10099 never arrived* · *SA-10301 has not shipped*")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            with st.expander("details"):
                st.json(m["meta"])

if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
    ][-6:]

    spinner = "Investigating..." if AGENT_MODE else "Thinking..."
    with st.chat_message("assistant"), st.spinner(spinner):
        try:
            if AGENT_MODE:
                r = httpx.post(
                    f"{API}/investigate", json={"complaint": prompt}, timeout=300
                )
            else:
                r = httpx.post(
                    f"{API}/chat", json={"message": prompt, "history": history}, timeout=120
                )
            if r.status_code == 429:
                st.warning("Rate limited - try again in a moment.")
                st.stop()
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

        st.markdown(data["answer"])

        if AGENT_MODE:
            # The trajectory is the point of the agentic mode -- show it.
            tags = [
                f"**{data['outcome']}**",
                f"{data['iterations']} iterations",
                f"{data['tokens'].get('total_tokens', 0)} tokens",
                f"{data['latency_ms']} ms",
            ]
            st.caption(" · ".join(tags))
            with st.expander(f"trajectory ({len(data['steps'])} steps)", expanded=True):
                for step in data["steps"]:
                    st.markdown(f"**{step['n']}.** `{step['tool']}` — {step['note']}")
                for i, v in enumerate(data["verifier_verdicts"], 1):
                    mark = "PASS" if v["passed"] else "REJECTED"
                    st.markdown(f"**verifier {i}:** {mark} — {'; '.join(v['problems']) or 'no problems'}")
            if data["findings"]:
                with st.expander("findings and citations"):
                    for f in data["findings"]:
                        st.markdown(f"- {f}")
                    st.caption("citations: " + (", ".join(data["citations"]) or "none"))
        else:
            tags = [f"`{data['intent']}`", f"{data['provider']}", f"{data['latency_ms']} ms"]
            if data["cached"]:
                tags.append("cached")
            if data["escalate"]:
                tags.append("escalated")
            st.caption(" · ".join(tags))
        with st.expander("details"):
            st.json(data)

    st.session_state.messages.append(
        {"role": "assistant", "content": data["answer"], "meta": data}
    )
