"""Streamlit tabanlı RAG arayüzü (çok turlu sohbet).

Çalıştırma:
    streamlit run app_streamlit.py
"""

import streamlit as st

from src.rag import answer_query

st.set_page_config(
    page_title="MC_RAG",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Özel stil (hem açık hem koyu temaya uyumlu) ---
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 780px;
        padding-top: 2rem;
    }
    h1 {
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    .mc-subtitle {
        color: var(--text-color);
        opacity: 0.6;
        font-size: 0.95rem;
        margin-top: -0.6rem;
        margin-bottom: 1.6rem;
    }
    .mc-badge-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
    }
    .mc-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        background: rgba(74, 157, 232, 0.14);
        background: color-mix(in srgb, #4a9de8 16%, transparent);
        color: #2f7bc9;
        color: color-mix(in srgb, #4a9de8 75%, var(--text-color));
        border: 1px solid rgba(74, 157, 232, 0.3);
        border: 1px solid color-mix(in srgb, #4a9de8 35%, transparent);
    }
    .mc-badge.mc-badge-time {
        background: rgba(52, 176, 110, 0.14);
        background: color-mix(in srgb, #34b06e 16%, transparent);
        color: #278552;
        color: color-mix(in srgb, #34b06e 75%, var(--text-color));
        border: 1px solid rgba(52, 176, 110, 0.3);
        border: 1px solid color-mix(in srgb, #34b06e 35%, transparent);
    }
    .mc-empty-state {
        text-align: center;
        padding: 3.5rem 1rem;
        color: var(--text-color);
        opacity: 0.5;
    }
    .mc-empty-state .mc-emoji {
        font-size: 2.4rem;
        margin-bottom: 0.6rem;
        opacity: 1;
    }
    .mc-empty-state .mc-hint {
        font-size: 0.85rem;
        margin-top: 0.4rem;
        opacity: 0.65;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔎 MC_RAG")
st.markdown(
    '<div class="mc-subtitle">Offline · Doküman tabanlı Soru-Cevap Asistanı</div>',
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{question, answer, sources, latency, chunks}]

# --- Kenar çubuğu: proje bilgisi + oturum özeti ---
with st.sidebar:
    st.markdown("### ℹ️ Hakkında")
    st.markdown(
        "MC_RAG, Microsoft Foundry Local üzerinde tamamen yerel çalışan bir "
        "Retrieval-Augmented Generation (RAG) asistanıdır. İnternet bağlantısı "
        "gerektirmez."
    )
    st.divider()
    st.markdown("### ⚙️ Nasıl çalışır?")
    st.markdown(
        "1. Sorunuz embedding'e dönüştürülür\n"
        "2. En alakalı doküman parçaları bulunur\n"
        "3. Bu parçalar bağlam olarak modele verilir\n"
        "4. Model, yalnızca bu bağlama dayanarak cevap üretir"
    )
    st.divider()

    turn_count = len(st.session_state.chat_history)
    if turn_count:
        avg_latency = sum(t["latency"] for t in st.session_state.chat_history) / turn_count
        col_a, col_b = st.columns(2)
        col_a.metric("Soru sayısı", turn_count)
        col_b.metric("Ort. süre", f"{avg_latency:.1f} sn")
    else:
        st.caption("Bu oturumda henüz soru soruşmadı.")


def _render_turn(question: str, answer: str, sources: list[str], latency: float, chunks: list) -> None:
    with st.chat_message("user", avatar="🧑"):
        st.write(question)
    with st.chat_message("assistant", avatar="🔎"):
        st.write(answer)

        source_badges = "".join(f'<span class="mc-badge">📄 {s}</span>' for s in sources) or (
            '<span class="mc-badge">📄 kaynak yok</span>'
        )
        st.markdown(
            f'<div class="mc-badge-row">{source_badges}'
            f'<span class="mc-badge mc-badge-time">⏱ {latency:.2f} sn</span></div>',
            unsafe_allow_html=True,
        )

        if chunks:
            with st.expander(f"Kullanılan {len(chunks)} chunk'ı gör"):
                for chunk in chunks:
                    st.markdown(
                        f"**{chunk.source_file}** · parça #{chunk.chunk_index} · "
                        f"benzerlik skoru `{chunk.score:.4f}`"
                    )
                    st.text(chunk.content)
                    st.divider()


# --- Sohbet geçmişi veya boş durum ---
if not st.session_state.chat_history:
    st.markdown(
        """
        <div class="mc-empty-state">
            <div class="mc-emoji">💬</div>
            <div>Henüz bir soru sormadınız.</div>
            <div class="mc-hint">Aşağıdaki kutuya yazarak başlayabilirsiniz.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for turn in st.session_state.chat_history:
        _render_turn(**turn)

question = st.chat_input("Sorunuzu yazın...")

if question:
    api_history = []
    for turn in st.session_state.chat_history:
        api_history.append({"role": "user", "content": turn["question"]})
        api_history.append({"role": "assistant", "content": turn["answer"]})

    try:
        with st.spinner("Cevap üretiliyor... (lütfen bekleyin, tıklamayın)"):
            result = answer_query(question, history=api_history)
    except Exception as exc:
        st.error(
            "Cevap üretilirken bir hata oluştu (muhtemelen işlem yarıda kesildi). "
            "Lütfen aynı soruyu tekrar gönderin.\n\n"
            f"Teknik detay: {exc}"
        )
        st.stop()

    _render_turn(question, result["answer"], result["sources"], result["latency"], result["chunks"])

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "latency": result["latency"],
            "chunks": result["chunks"],
        }
    )

if st.session_state.chat_history:
    st.divider()
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🗑️ Temizle", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()