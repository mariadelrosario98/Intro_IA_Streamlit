import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import concurrent.futures
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
groq_api_key = st.sidebar.text_input("🔑 Ingresa tu API Key de Groq (gsk_...):", type="password")
temperature = st.sidebar.slider("Temperatura del Modelo", 0.0, 1.0, 0.7, 0.1)

# Modelo fijo recomendado (Groq)
MODEL_NAME = "llama-3.1-70b"   # ✅ nombre correcto

# --- Funciones ---

def load_llm(api_key, temperature, model_name=MODEL_NAME):
    """Inicializa y retorna LLaMA-3.1-70B en Groq."""
    if not api_key:
        st.error("Por favor, ingresa tu API Key de Groq en el sidebar.")
        return None
    try:
        llm = ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature
        )
        return llm
    except Exception as e:
        st.error(f"Error al cargar el modelo Groq: {e}")
        return None


def run_no_rag_query(query, api_key, temperature, model_name=MODEL_NAME):
    """Consulta directa sin RAG."""
    llm = load_llm(api_key, temperature, model_name)
    if not llm:
        return "Error: No se pudo cargar el modelo."
    prompt_template = PromptTemplate(
        input_variables=["pregunta"],
        template="Eres un experto en agricultura. Responde de manera concisa a la siguiente pregunta: {pregunta}"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    try:
        return chain.run(query)
    except Exception as e:
        return f"Error en la ejecución sin RAG: {e}"


def run_rag_query(query, api_key, temperature, model_name=MODEL_NAME):
    """RAG simplificado: buscar en Wikipedia y pasar resumen al modelo."""
    llm = load_llm(api_key, temperature, model_name)
    if not llm:
        return "Error: No se pudo cargar el modelo para RAG."

    try:
        wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
        context = wiki.run(query)
        enriched_prompt = f"Contexto de Wikipedia:\n{context}\n\nPregunta: {query}\nRespuesta:"
        
        prompt_template = PromptTemplate(
            input_variables=["pregunta"],
            template="{pregunta}"
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        return chain.run(enriched_prompt)
    except Exception as e:
        return f"Error en RAG: {e}"


def run_csv_query(query, df, api_key, temperature, model_name=MODEL_NAME):
    """Responde preguntas en contexto usando el CSV subido."""
    llm = load_llm(api_key, temperature, model_name)
    if not llm:
        return "Error: No se pudo cargar el modelo."

    # Convertimos las primeras filas a texto
    preview = df.head(20).to_string()

    enriched_prompt = f"""
    Tienes acceso a un dataset de agricultura con estas columnas: {', '.join(df.columns)}.
    Aquí van las primeras filas como referencia:
    {preview}

    Ahora responde a la pregunta del usuario, usando solo la información del dataset:
    {query}
    """

    prompt_template = PromptTemplate(
        input_variables=["pregunta"],
        template="{pregunta}"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(enriched_prompt)

# ======================
# App principal
# ======================

st.title("🚜 Herramienta de Análisis y QA sobre Agricultura")
st.subheader("Análisis Exploratorio de Datos (EDA) y Comparación de Respuestas")

# --- Sección EDA ---
st.header("1. Exploración de Datos (EDA) de Agricultura")
uploaded_file = st.file_uploader("Sube un archivo CSV sobre agricultura", type="csv")

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Archivo cargado exitosamente.")
        st.write("---")
        
        st.subheader("📊 Vista previa de los datos")
        st.dataframe(df.head())
        
        st.subheader("ℹ️ Información del DataFrame")
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        st.subheader("📈 Estadísticas Descriptivas")
        st.write(df.describe())
        
        st.subheader("📉 Matriz de Correlación")
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Matriz de Correlación")
            st.pyplot(plt)
        else:
            st.warning("No hay columnas numéricas para calcular la matriz de correlación.")
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")

# --- Sección Preguntas ---
st.write("---")
st.header("2. Comparación de Respuestas (paralelo)")
st.info("Se generarán tres respuestas en paralelo: sin RAG, con RAG (Wikipedia) y con datos del CSV.")

user_question = st.text_area("Haz una pregunta sobre agricultura:", "ej: ¿Cuáles son los beneficios de la rotación de cultivos?")

if st.button("Obtener Respuestas"):
    if not groq_api_key:
        st.warning("Por favor, ingresa tu API Key de Groq en el sidebar para continuar.")
    else:
        with st.spinner("Generando respuestas en paralelo..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_no_rag = executor.submit(run_no_rag_query, user_question, groq_api_key, temperature, MODEL_NAME)
                future_rag = executor.submit(run_rag_query, user_question, groq_api_key, temperature, MODEL_NAME)
                if df is not None:
                    future_csv = executor.submit(run_csv_query, user_question, df, groq_api_key, temperature, MODEL_NAME)
                else:
                    future_csv = None

                response_no_rag = future_no_rag.result()
                response_rag = future_rag.result()
                response_csv = future_csv.result() if future_csv else "No se ha cargado ningún CSV."

            st.session_state.llm_no_rag = response_no_rag
            st.session_state.llm_rag = response_rag
            st.session_state.llm_csv = response_csv

# --- Comparación lado a lado ---
if "llm_no_rag" in st.session_state and "llm_rag" in st.session_state and "llm_csv" in st.session_state:
    st.write("---")
    st.header("3. Resultados lado a lado")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Sin RAG")
        st.info(st.session_state.llm_no_rag)
        
    with col2:
        st.subheader("Con RAG (Wikipedia)")
        st.info(st.session_state.llm_rag)

    with col3:
        st.subheader("Con CSV")
        st.info(st.session_state.llm_csv)
