import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import os
from langchain_community.llms import HuggingFaceEndpoint   # ✅ solo usamos esta
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv

# Load environment variables (opcional si usas .env localmente)
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="Herramienta de Análisis y QA sobre Agricultura",
    page_icon="🌱",
    layout="wide"
)

# --- Sidebar ---
st.sidebar.title("🛠️ Configuración")
hf_token = st.sidebar.text_input("Ingresa tu Secret de Hugging Face:", type="password")
temperature = st.sidebar.slider("Temperatura del Modelo", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# --- Functions ---

def load_llm(hf_token, temperature):
    """Initializes and returns a Hugging Face Hub LLM."""
    if not hf_token:
        st.error("Por favor, ingresa tu Secret de Hugging Face en el sidebar.")
        return None
    try:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=temperature,
            max_new_tokens=512
        )
        return llm
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def run_rag_query(query, hf_token, temperature):
    """Runs a query with RAG using Wikipedia as the source."""
    llm = load_llm(hf_token, temperature)
    if not llm:
        return "Error: No se pudo cargar el modelo para RAG."

    wikipedia_tool = WikipediaAPIWrapper()
    tools = [Tool(
        name="Wikipedia",
        func=wikipedia_tool.run,
        description="Útil para buscar información en Wikipedia."
    )]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False
    )
    
    try:
        response = agent.run(f"Responde la siguiente pregunta sobre agricultura, usando solo fuentes de Wikipedia: {query}")
        return response
    except Exception as e:
        st.error(f"Error en la ejecución del agente RAG: {e}")
        return "Error al procesar la solicitud con RAG. Asegúrate de que tu token es válido y la API está disponible."

def run_no_rag_query(query, hf_token, temperature):
    """Runs a query without RAG, directly with the LLM."""
    llm = load_llm(hf_token, temperature)
    if not llm:
        return "Error: No se pudo cargar el modelo."

    prompt_template = PromptTemplate(
        input_variables=["pregunta"],
        template="Eres un experto en agricultura. Responde de manera concisa a la siguiente pregunta: {pregunta}"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    try:
        response = chain.run(query)
        return response
    except Exception as e:
        st.error(f"Error en la ejecución del modelo sin RAG: {e}")
        return "Error al procesar la solicitud sin RAG."

# --- Main App ---
st.title("🚜 Herramienta de Análisis y QA sobre Agricultura")
st.subheader("Análisis Exploratorio de Datos (EDA) y Agente de Preguntas sobre Agricultura")

# --- EDA Section ---
st.header("1. Exploración de Datos (EDA) de Agricultura")
uploaded_file = st.file_uploader("Sube un archivo CSV sobre agricultura", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success("Archivo cargado exitosamente.")
        st.write("---")
        
        # 1. Muestra del DataFrame
        st.subheader("📊 Vista previa de los datos")
        st.dataframe(df.head())
        
        # 2. Información General
        st.subheader("ℹ️ Información del DataFrame")
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        # 3. Estadísticas Descriptivas
        st.subheader("📈 Estadísticas Descriptivas")
        st.write(df.describe())
        
        # 4. Matriz de Correlación
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

# --- Questions Section ---
st.write("---")
st.header("2. Agente de Preguntas sobre Agricultura")
st.info("Este agente solo responde preguntas relacionadas con la agricultura. No podrá responder otras preguntas.")

user_question = st.text_area("Haz una pregunta sobre agricultura:", "ej: ¿Cuáles son los beneficios de la rotación de cultivos?")

if st.button("Obtener Respuesta"):
    if not hf_token:
        st.warning("Por favor, ingresa tu Secret de Hugging Face en el sidebar para continuar.")
    else:
        with st.spinner("Generando respuesta sin RAG..."):
            response_no_rag = run_no_rag_query(user_question, hf_token, temperature)
        
        with st.spinner("Generando respuesta con RAG..."):
            response_rag = run_rag_query(user_question, hf_token, temperature)
        
        # Store responses in session state for later display
        st.session_state.llm_no_rag = response_no_rag
        st.session_state.llm_rag = response_rag

# --- Comparison Tab Section ---
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
