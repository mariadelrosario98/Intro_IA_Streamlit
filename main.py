import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from huggingface_hub import InferenceClient   # ✅ cliente oficial
from langchain_community.utilities import WikipediaAPIWrapper

# ======================
# Configuración de página
# ======================
st.set_page_config(
    page_title="Herramienta de Análisis y QA sobre Agricultura",
    page_icon="🌱",
    layout="wide"
)

# --- Sidebar ---
st.sidebar.title("🛠️ Configuración")
hf_token = st.sidebar.text_input("Ingresa tu Secret de Hugging Face:", type="password")
temperature = st.sidebar.slider("Temperatura del Modelo", 0.0, 1.0, 0.7, 0.1)

# --- Funciones ---

def query_hf_model(hf_token, question, temperature=0.7):
    """Consulta directa al modelo de Hugging Face vía chat_completion."""
    if not hf_token:
        return "Error: No se proporcionó token de Hugging Face."
    try:
        client = InferenceClient(
            "mistralai/Mistral-7B-Instruct-v0.2",
            token=hf_token
        )
        response = client.chat_completion(
            messages=[{"role": "user", "content": question}],
            max_tokens=512,
            temperature=temperature
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error al consultar el modelo: {e}"

def run_rag_query(question, hf_token, temperature):
    """RAG manual: busca contexto en Wikipedia y lo pasa al modelo."""
    try:
        wiki = WikipediaAPIWrapper()
        context = wiki.run(question)
        enriched_prompt = f"Contexto de Wikipedia:\n{context}\n\nPregunta: {question}\nRespuesta:"
        return query_hf_model(hf_token, enriched_prompt, temperature)
    except Exception as e:
        return f"Error en RAG: {e}"

# ======================
# App principal
# ======================

st.title("🚜 Herramienta de Análisis y QA sobre Agricultura")
st.subheader("Análisis Exploratorio de Datos (EDA) y Agente de Preguntas sobre Agricultura")

# --- Sección EDA ---
st.header("1. Exploración de Datos (EDA) de Agricultura")
uploaded_file = st.file_uploader("Sube un archivo CSV sobre agricultura", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success("Archivo cargado exitosamente.")
        st.write("---")
        
        # Vista previa
        st.subheader("📊 Vista previa de los datos")
        st.dataframe(df.head())
        
        # Información general
        st.subheader("ℹ️ Información del DataFrame")
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        # Estadísticas descriptivas
        st.subheader("📈 Estadísticas Descriptivas")
        st.write(df.describe())
        
        # Matriz de correlación
        st.subheader("📉 Matriz de Correlación")
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Matriz de Correlación")
            st.pyplot(plt)
            st.write("Esta matriz muestra la correlación entre las variables numéricas. Valores cercanos a 1 o -1 indican una fuerte correlación positiva o negativa, respectivamente.")
        else:
            st.warning("No hay columnas numéricas para calcular la matriz de correlación.")
            
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")

# --- Sección Preguntas ---
st.write("---")
st.header("2. Agente de Preguntas sobre Agricultura")
st.info("Este agente solo responde preguntas relacionadas con la agricultura. No podrá responder otras preguntas.")

user_question = st.text_area("Haz una pregunta sobre agricultura:", "ej: ¿Cuáles son los beneficios de la rotación de cultivos?")

if st.button("Obtener Respuesta"):
    if not hf_token:
        st.warning("Por favor, ingresa tu Secret de Hugging Face en el sidebar para continuar.")
    else:
        with st.spinner("Generando respuesta sin RAG..."):
            response_no_rag = query_hf_model(hf_token, user_question, temperature)

        with st.spinner("Generando respuesta con RAG..."):
            response_rag = run_rag_query(user_question, hf_token, temperature)

        st.session_state.llm_no_rag = response_no_rag
        st.session_state.llm_rag = response_rag

# --- Comparación ---
st.write("---")
st.header("3. Comparación de Respuestas")
if "llm_no_rag" in st.session_state and st.session_state.llm_no_rag and "llm_rag" in st.session_state and st.session_state.llm_rag:
    tab1, tab2 = st.tabs(["Sin RAG", "Con RAG"])
    
    with tab1:
        st.subheader("Respuesta sin RAG (solo el modelo)")
        st.info(st.session_state.llm_no_rag)
        st.write("Esta respuesta se genera directamente por el modelo de lenguaje, basándose en la información con la que fue entrenado.")
        
    with tab2:
        st.subheader("Respuesta con RAG (fuente externa)")
        st.info(st.session_state.llm_rag)
        st.write("Esta respuesta utiliza la **Generación Aumentada por Recuperación (RAG)** para buscar información relevante en fuentes externas (como Wikipedia) antes de generar la respuesta. Esto puede ayudar a proporcionar información más actualizada y precisa.")
