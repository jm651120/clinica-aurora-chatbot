"""
app.py — Streamlit chat interface for the Clínica Aurora RAG assistant.

Run with:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from src.retrieval import load_vectorstore, retrieve, format_context
from src.generation import generate_response, rewrite_query
from src.postprocessing import sanitize as sanitize_output
from src.pack_comparison import is_pack_query, build_pack_context

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistente — Clínica Aurora",
    page_icon="✨",
    layout="centered",
)

# ── Load vector store once per session ────────────────────────────────────────
@st.cache_resource(show_spinner="A carregar base de conhecimento …")
def get_vectorstore():
    try:
        return load_vectorstore()
    except FileNotFoundError as e:
        return None, str(e)


vectorstore = get_vectorstore()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✨ Clínica Aurora")
    st.markdown("**Medicina Estética em Lisboa**")
    st.divider()
    st.markdown("""
📍 Av. da Liberdade, 215 — Lisboa
📞 21 345 67 89
📧 info@clinicaaurora.pt
🌐 www.clinicaaurora.pt
    """)
    st.divider()
    st.markdown("**Horário**")
    st.markdown("Seg–Sex: 9h–20h | Sáb: 9h–17h")
    st.divider()

    show_sources = st.toggle("Mostrar fontes utilizadas", value=False)

    if st.button("Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Assistente Virtual — Clínica Aurora")
st.caption("Olá! Sou o assistente virtual da Clínica Aurora. "
           "Posso ajudá-lo/a com informações sobre os nossos tratamentos, "
           "preços, marcações e políticas.")

# Block if vector store is missing
if vectorstore is None:
    st.error(
        "⚠️ Base de conhecimento não encontrada. "
        "Execute primeiro: `python -m src.ingestion`"
    )
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg.get("sources"):
            with st.expander("Fontes consultadas"):
                st.markdown(msg["sources"])

# ── Handle new user input ─────────────────────────────────────────────────────
if prompt := st.chat_input("Escreva a sua pergunta aqui …"):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve → Augment → Generate → Sanitize
    with st.chat_message("assistant"):
        with st.spinner("A pesquisar e a preparar resposta …"):
            expanded_query = rewrite_query(prompt)
            docs = retrieve(expanded_query, vectorstore)
            context = format_context(docs)

            # Prepend pre-computed pack data when the query involves a pack
            # comparison. The LLM receives authoritative numbers and is
            # instructed (rule 6) to format them rather than recalculate.
            if is_pack_query(prompt) or is_pack_query(expanded_query):
                context = build_pack_context("pack_noiva") + "\n\n" + context

            answer = generate_response(
                query=prompt,
                context=context,
                history=st.session_state.messages[:-1],  # exclude current turn
            )
            answer = sanitize_output(answer)

        st.markdown(answer)

        # Format sources for optional display
        source_text = ""
        if show_sources and expanded_query != prompt:
            source_text += f"*Query expandida: `{expanded_query}`*\n\n"
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "desconhecido")
            snippet = doc.page_content[:120].replace("\n", " ").strip()
            source_text += f"**[{i}]** `{src}` — {snippet}…\n\n"

        if show_sources and source_text:
            with st.expander("Fontes consultadas"):
                st.markdown(source_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_text,
    })
