import streamlit as st
import requests
import time

st.set_page_config(page_title="Chat Legal UY", layout="centered")
st.title("💬 Asistente Legal Uruguay (RAG + GPT-3.5-turbo)")

with st.sidebar:
    st.header("⚖️ Chat Legal UY")
    st.markdown(
        """
        - Escribe tu consulta legal en el chat.
        - Las respuestas se generan con RAG y GPT-3.5-turbo.
        - El modelo solo responde usando el contexto legal cargado.
        """
    )
    api_url = st.text_input(
        "URL de la API",
        value="http://localhost:8000/chat",
        help="URL del endpoint FastAPI /chat"
    )
    st.markdown("---")
    if st.button("🧹 Limpiar chat"):
        st.session_state.messages = []
        st.rerun()
    st.info("Desarrollado por Felipe, Valentina y Natalia")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial de mensajes (de arriba hacia abajo)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Input de usuario (queda abajo y centrado con el diseño default de Streamlit)
if prompt := st.chat_input("Escribe tu mensaje..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    # Llamada al backend con streaming
    with st.chat_message("assistant"):
        start_time = time.time()
        placeholder = st.empty()
        response_text = ""
        try:
            with st.spinner("Pensando..."):
                with requests.post(api_url, json={"prompt": prompt}, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    # Lee la respuesta en streaming y la muestra como markdown
                    for chunk in r.iter_content(chunk_size=512):
                        if chunk:
                            text = chunk.decode("utf-8")
                            response_text += text
                            placeholder.markdown(response_text, unsafe_allow_html=True)
            elapsed = time.time() - start_time
            st.caption(f"⏱️ Tiempo de respuesta: {elapsed:.2f} segundos")
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})