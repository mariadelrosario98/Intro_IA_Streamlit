import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import random

# --- CONFIGURACIÓN DE LA PÁGINA Y DEL LOGO ---
logo_path = "logo.png"
logo_exists = os.path.exists(logo_path)

st.set_page_config(
    page_title="Bio Gemini",
    page_icon=logo_path if logo_exists else "🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- INYECCIÓN DE CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

:root {
    --primary-bg: #1e1e1e;
    --secondary-bg: #2b2b2b;
    --user-message-bg: #3c4043;
    --accent-green: #79F8B7;
    --accent-green-dark: #00A67E;
    --text-light: #FFFFFF; /* BLANCO PURO */
    --text-dark: #000000;
    --border-color: #4a4d52;
}

body { font-family: 'Google Sans', sans-serif, system-ui; background-color: var(--primary-bg) !important; color: var(--text-light); }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stBottomBlockContainer"] { background-color: var(--primary-bg); }
[data-testid="stSidebar"], [data-testid="stHeader"], #MainMenu, footer { display: none; }

div[data-testid="stForm"] button { background-color: var(--accent-green-dark); color: white; font-weight: 600; border: none; border-radius: 8px; padding: 12px 20px; width: 100%; transition: background-color 0.3s ease; }
div[data-testid="stForm"] button:hover { background-color: var(--accent-green); color: var(--primary-bg); border: none; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
[data-testid="stChatMessage"] { background: none; padding: 0; margin: 0; animation: fadeIn 0.5s ease-out; }
.stChatMessage { max-width: 768px; margin: 0 auto 2rem auto; }
[data-testid="stChatMessage"] [data-testid="stAvatar"] img { width: 40px; height: 40px; }
.st-emotion-cache-1c7y2kd { color: var(--text-light); font-weight: 600; padding-bottom: 8px; }

/* <<--- CORRECCIÓN DE MÁXIMA PRIORIDAD PARA EL COLOR DEL TEXTO --->> */

/* Regla general para el contenedor del mensaje */
[data-testid="stChatMessage"] > div[data-testid="stMarkdown"] {
    padding: 1.2em;
    border-radius: 12px;
    line-height: 1.6;
}

/* Reglas específicas y forzadas para el texto del AGENTE */
[data-testid="stChatMessage"][data-testid="chat-message-assistant"] > div[data-testid="stMarkdown"],
[data-testid="stChatMessage"][data-testid="chat-message-assistant"] p,
[data-testid="stChatMessage"][data-testid="chat-message-assistant"] li,
[data-testid="stChatMessage"][data-testid="chat-message-assistant"] strong {
    color: #FFFFFF !important; /* FORZAR TEXTO BLANCO */
}

/* Reglas específicas y forzadas para el texto del USUARIO */
[data-testid="stChatMessage"][data-testid="chat-message-user"] > div[data-testid="stMarkdown"],
[data-testid="stChatMessage"][data-testid="chat-message-user"] p,
[data-testid="stChatMessage"][data-testid="chat-message-user"] li,
[data-testid="stChatMessage"][data-testid="chat-message-user"] strong {
    color: #FFFFFF !important; /* FORZAR TEXTO BLANCO */
}

/* Fondos de las burbujas */
[data-testid="stChatMessage"][data-testid="chat-message-assistant"] > div[data-testid="stMarkdown"] {
    background-color: var(--secondary-bg);
    border: 1px solid var(--border-color);
}
[data-testid="stChatMessage"][data-testid="chat-message-user"] > div[data-testid="stMarkdown"] {
    background-color: var(--user-message-bg);
    border: 1px solid var(--border-color);
}

/* Input de Texto */
[data-testid="stChatInput"] { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background-color: var(--primary-bg); border-top: 1px solid var(--border-color); padding-top: 10px; width: 100%; max-width: 768px; }
[data-testid="stChatInput"] textarea { background-color: #FFFFFF; color: var(--text-dark); border-radius: 18px; border: 1px solid #DDDDDD; }
[data-testid="stChatInput"] textarea::placeholder { color: #666666; }
</style>
""", unsafe_allow_html=True)


# --- LÓGICA DEL AGENTE CON MEMORIA ---
@st.cache_resource
def get_conversation_chain(_api_key):
    PROMPT = ChatPromptTemplate.from_messages([
        ("system", """Eres 'Bio Gemini', un agente de IA experto en biología. Responde de forma precisa, educativa y amigable. Utiliza el historial de conversación para entender el contexto de las nuevas preguntas.
        Reglas:
        - No des consejos médicos o veterinarios.
        - Usa **negritas** para términos clave.
        - Al final de cada explicación, sugiere de forma natural una pregunta de seguimiento interesante y relacionada con el tema para fomentar la curiosidad del usuario.
        """),
        ("human", "{history}"),
        ("human", "{input}")
    ])
    llm = ChatGroq(api_key=_api_key, model="llama3-70b-8192")
    memory = ConversationBufferMemory(memory_key="history", input_key="input")
    conversation_chain = ConversationChain(llm=llm, prompt=PROMPT, verbose=False, memory=memory)
    return conversation_chain

# --- GESTIÓN DE ESTADO DE SESIÓN ---
if "groq_api_key" not in st.session_state: st.session_state.groq_api_key = None
if "messages" not in st.session_state: st.session_state.messages = []
if "chain" not in st.session_state: st.session_state.chain = None

# --- PANTALLA DE BIENVENIDA / API KEY ---
if not st.session_state.groq_api_key:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if logo_exists:
            img_col1, img_col2, img_col3 = st.columns([1,1,1])
            with img_col2: st.image(logo_path, width=80)
        else:
            st.markdown("<p style='font-size: 60px; text-align: center;'>🧬</p>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #FFFFFF;'>Bienvenido a Bio Gemini</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; opacity: 0.8;'>Para comenzar, ingresa tu API Key de Groq.</p>", unsafe_allow_html=True)
        with st.form("api_key_form"):
            api_key_input = st.text_input("Tu API Key de Groq", type="password", placeholder="gsk_xxxxxxxxxx", help="Obtén tu clave gratuita en console.groq.com", label_visibility="collapsed")
            submitted = st.form_submit_button("Activar Bio Gemini")
            if submitted:
                if api_key_input and api_key_input.startswith("gsk_"):
                    st.session_state.groq_api_key = api_key_input
                    st.session_state.chain = get_conversation_chain(st.session_state.groq_api_key)
                    st.rerun()
                else:
                    st.error("Por favor, ingresa una API Key de Groq válida.")

# --- INTERFAZ DE CHAT PRINCIPAL ---
else:
    col1, col2, col3 = st.columns([2,3,2])
    with col1:
        st.markdown('<div style="text-align: left; padding-top: 1rem;">', unsafe_allow_html=True)
        if st.button("➕ Nuevo Chat"):
            st.session_state.messages = []
            if st.session_state.chain: st.session_state.chain.memory.clear()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='text-align: center; color: var(--accent-green);'>Bio Gemini</h1>", unsafe_allow_html=True)

    USER_AVATAR = "👤"
    BOT_AVATAR = logo_path if logo_exists else "✨"

    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Hola, soy Bio Gemini. ¿En qué puedo ayudarte hoy?"})

    for message in st.session_state.messages:
        avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pregúntame algo de biología..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            try:
                def extract_response_from_stream(stream):
                    for chunk in stream:
                        if "response" in chunk:
                            yield chunk["response"]
                
                response_stream = st.session_state.chain.stream({"input": prompt})
                response_content = st.write_stream(extract_response_from_stream(response_stream))
                st.session_state.messages.append({"role": "assistant", "content": response_content})

            except Exception as e:
                error_message = "Lo siento, ha ocurrido un error. Verifica tu API Key o tu conexión."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
        
        st.rerun()
    with tab2:
        st.subheader("Respuesta con RAG (fuente externa)")
        st.info(st.session_state.llm_rag)
        st.write("Esta respuesta utiliza la **Generación Aumentada por Recuperación (RAG)** para buscar información relevante en fuentes externas (como Wikipedia) antes de generar la respuesta. Esto puede ayudar a proporcionar información más actualizada y precisa.")
