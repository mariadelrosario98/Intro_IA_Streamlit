import pandas as pd
import streamlit as st
import plotly.express as px
from huggingface_hub import InferenceClient
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain import hub
from langchain_community.llms import HuggingFaceHub
from langchain_community.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# --- Configuración de la página ---
st.set_page_config(page_title="Análisis de Agricultura Colombiana y Agente IA 🚜", layout="wide")

# --- Variables globales ---
if 'df' not in st.session_state:
    st.session_state['df'] = None
    
# --- Funciones de EDA ---
def perform_eda(df):
    st.header("Análisis Exploratorio de Datos (EDA) 📊")
    st.write(f"El conjunto de datos contiene {df.shape[0]} filas y {df.shape[1]} columnas.")
    
    st.subheader("Estadísticas Descriptivas")
    st.write(df.describe(include='all'))
    
    st.subheader("Visualizaciones")
    numerics = df.select_dtypes(include=['float64', 'int64']).columns
    if not numerics.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.write("Distribución de los 10 cultivos más importantes por área:")
            area_df = df.dropna(subset=['Área Cosechada (ha)']).sort_values(by='Área Cosechada (ha)', ascending=False).head(10)
            fig_bar = px.bar(area_df, x='Cultivo', y='Área Cosechada (ha)', title="Cultivos con Mayor Área Cosechada")
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            st.write("Distribución de la producción por departamento:")
            depto_df = df.dropna(subset=['Producción (t)']).groupby('Departamento')['Producción (t)'].sum().reset_index()
            fig_pie = px.pie(depto_df, values='Producción (t)', names='Departamento', title="Producción Total por Departamento")
            st.plotly_chart(fig_pie, use_container_width=True)
            
    st.subheader("Mapa de Correlación")
    try:
        corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, title="Mapa de Correlación")
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.error(f"No se pudo generar el mapa de correlación. Error: {e}")

# --- Agente de IA sin RAG ---
def create_agent_without_rag(token):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature": 0.5, "max_length": 64})
    
    @tool
    def query_dataframe(query: str):
        """Usa esta herramienta para consultar el dataframe sobre agricultura.
           Puedes hacer preguntas como: '¿Cuál es el cultivo más producido en Cundinamarca?',
           '¿Qué departamento tiene la mayor área cosechada de maíz?'."""
        try:
            return eval("st.session_state['df']" + query)
        except Exception as e:
            return f"Error al ejecutar la consulta: {e}. Asegúrate de que tu consulta sea válida."

    tools = [query_dataframe]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor

# --- Agente de IA con RAG ---
def create_agent_with_rag(token):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature": 0.5, "max_length": 64})
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_text = st.session_state['df'].to_string(index=False)
    docs = text_splitter.create_documents([doc_text])
    
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# --- Interfaz de usuario ---
st.sidebar.title("Configuración ⚙️")
huggingface_token = st.sidebar.text_input("Pega tu Secret de Hugging Face aquí", type="password")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV de agricultura", type="csv")

if uploaded_file:
    st.session_state['df'] = pd.read_csv(uploaded_file, encoding='latin-1')
    st.session_state['df'].columns = [col.strip() for col in st.session_state['df'].columns]
    st.sidebar.success("Archivo cargado exitosamente.")
    
tab1, tab2 = st.tabs(["Análisis de Datos (EDA) 📈", "Agente de IA 🤖"])

with tab1:
    if st.session_state['df'] is not None:
        perform_eda(st.session_state['df'])
    else:
        st.info("Sube un archivo CSV en la barra lateral para comenzar el análisis.")

with tab2:
    if not huggingface_token:
        st.warning("Pega tu secret de Hugging Face en la barra lateral para activar el agente.")
    elif st.session_state['df'] is None:
        st.info("Sube un archivo CSV para que el agente tenga datos sobre los que responder.")
    else:
        st.subheader("Elige el tipo de Agente")
        agent_type = st.radio("Selecciona el modo del agente", ("Con RAG", "Sin RAG"))
        
        st.subheader("Pregúntale al Agente")
        user_query = st.text_input("Ingresa tu pregunta sobre los datos de agricultura:")
        
        if st.button("Obtener Respuesta"):
            if user_query:
                with st.spinner("Generando respuesta..."):
                    try:
                        if agent_type == "Con RAG":
                            qa_chain = create_agent_with_rag(huggingface_token)
                            response = qa_chain({"query": user_query})
                            st.write("### Respuesta del Agente:")
                            st.write(response["result"])
                        else:
                            agent_executor = create_agent_without_rag(huggingface_token)
                            response = agent_executor.invoke({"input": user_query})
                            st.write("### Respuesta del Agente:")
                            st.write(response["output"])
                    except Exception as e:
                        st.error(f"Ocurrió un error al generar la respuesta: {e}. Asegúrate de que el token sea válido y el modelo esté accesible.")
