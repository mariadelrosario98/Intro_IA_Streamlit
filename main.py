import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import concurrent.futures
import pydeck as pdk
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.agents import create_pandas_dataframe_agent

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

MODEL_NAME = "llama-3.1-8b-instant"
temperature = st.sidebar.slider("Temperatura del Modelo", 0.0, 1.0, 0.7, 0.1)

# ======================
# Funciones auxiliares
# ======================

def load_llm(api_key, temperature, model_name=MODEL_NAME):
    if not api_key:
        st.error("Por favor, ingresa tu API Key de Groq en el sidebar.")
        return None
    try:
        return ChatGroq(api_key=api_key, model_name=model_name, temperature=temperature)
    except Exception as e:
        st.error(f"Error al cargar el modelo Groq: {e}")
        return None

def run_no_rag_query(query, api_key, temperature):
    llm = load_llm(api_key, temperature)
    if not llm:
        return "Error: No se pudo cargar el modelo."
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto en agricultura. Responde de manera concisa."),
        ("user", "{pregunta}")
    ])
    chain = prompt | llm
    try:
        return chain.invoke({"pregunta": query}).content
    except Exception as e:
        return f"Error en la ejecución sin RAG: {e}"

def run_rag_query(query, api_key, temperature):
    llm = load_llm(api_key, temperature)
    if not llm:
        return "Error: No se pudo cargar el modelo para RAG."
    try:
        wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
        context = wiki.run(query)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres un experto en agricultura. Usa el contexto de Wikipedia para responder."),
            ("user", f"Contexto: {context}\n\nPregunta: {query}")
        ])
        chain = prompt | llm
        return chain.invoke({}).content
    except Exception as e:
        return f"Error en RAG: {e}"

def run_csv_agent_query(query, df, api_key, temperature):
    llm = load_llm(api_key, temperature)
    if not llm:
        return "Error: No se pudo cargar el modelo."
    try:
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )
        response = agent.run(query)

        # Capturar gráficos automáticos
        fig = plt.gcf()
        if fig and fig.get_axes():
            st.pyplot(fig)
            plt.clf()

        return response
    except Exception as e:
        return f"Error en el agente CSV: {e}"

# ======================
# App principal
# ======================

st.title("🚜 Herramienta de Análisis y QA sobre Agricultura")
st.subheader("Exploración de Datos (EDA) y Comparación de Respuestas")

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
        st.text(buffer.getvalue())

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

        # --- NUEVA PARTE: mapa interactivo ---
        lat_candidates = [c for c in df.columns if "lat" in c.lower()]
        lon_candidates = [c for c in df.columns if "lon" in c.lower()]

        if lat_candidates and lon_candidates:
            lat_col, lon_col = lat_candidates[0], lon_candidates[0]

            st.subheader("🌍 Mapa de fincas por cultivo")
            cultivos = df['cultivo'].dropna().unique().tolist()
            cultivo_sel = st.selectbox("Selecciona un cultivo:", ["Todos"] + cultivos)

            df_map = df if cultivo_sel == "Todos" else df[df['cultivo'] == cultivo_sel]

            if not df_map.empty:
                # Forzar centro en Colombia si hay valores inválidos
                try:
                    lat_center = df_map[lat_col].mean()
                    lon_center = df_map[lon_col].mean()
                    if pd.isna(lat_center) or pd.isna(lon_center):
                        lat_center, lon_center = 4.5709, -74.2973
                except Exception:
                    lat_center, lon_center = 4.5709, -74.2973

                # Colores distintos por cultivo
                df_map["color"] = df_map["cultivo"].astype("category").cat.codes * 40 + 60

                st.pydeck_chart(pdk.Deck(
                    map_style=None,  # ✅ usa OpenStreetMap (sin token externo)
                    initial_view_state=pdk.ViewState(
                        latitude=lat_center,
                        longitude=lon_center,
                        zoom=5,  # 🔥 zoom nacional
                        pitch=0,
                    ),
                    layers=[
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=df_map,
                            get_position=f'[{lon_col}, {lat_col}]',
                            get_color='[color, 100, 200]',
                            get_radius=20000,
                            pickable=True
                        )
                    ],
                    tooltip={
                        "html": "<b>Cultivo:</b> {cultivo}<br/>"
                                + ("<b>Región:</b> {region}<br/>" if "region" in df.columns else "")
                                + "<b>Rendimiento:</b> {rendimiento_t_ha}",
                        "style": {"color": "white"}
                    }
                ))
            else:
                st.warning("No hay datos para ese cultivo seleccionado.")
        else:
            st.warning("⚠️ No se encontraron columnas de lat/lon en el CSV.")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")

# --- Sección Preguntas ---
st.write("---")
st.header("2. Comparación de Respuestas (paralelo)")
st.info("Respuestas en paralelo: sin RAG, con RAG (Wikipedia) y con agente multipropósito sobre CSV.")

user_question = st.text_area("Haz una pregunta sobre agricultura:", "ej: ¿Cuál región produce más maíz y papa?")

if st.button("Obtener Respuestas"):
    if not groq_api_key:
        st.warning("Por favor, ingresa tu API Key de Groq en el sidebar para continuar.")
    else:
        with st.spinner("Generando respuestas en paralelo..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_no_rag = executor.submit(run_no_rag_query, user_question, groq_api_key, temperature)
                future_rag = executor.submit(run_rag_query, user_question, groq_api_key, temperature)
                if df is not None:
                    future_csv = executor.submit(run_csv_agent_query, user_question, df, groq_api_key, temperature)
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
        st.subheader("Con CSV (Agente multipropósito)")
        st.info(st.session_state.llm_csv)
