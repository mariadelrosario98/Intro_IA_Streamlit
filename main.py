# Este será el código de Python
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date

# -----------------------------
# Configuración de la página
# -----------------------------
st.set_page_config(page_title="EDA con datos aleatorios", page_icon="📊", layout="wide")
st.title("📊 EDA con datos aleatorios")
st.caption("Demostración rápida en Streamlit con visualizaciones de línea y barras")

# -----------------------------
# Sidebar (controles)
# -----------------------------
st.sidebar.header("Parámetros de los datos")
seed = st.sidebar.number_input("Semilla aleatoria", min_value=0, max_value=10_000, value=42, step=1)
n_dias = st.sidebar.slider("Número de días", min_value=15, max_value=365, value=90, step=5)
n_categorias = st.sidebar.slider("Número de categorías", min_value=2, max_value=10, value=5, step=1)
media = st.sidebar.number_input("Media de la distribución", value=100.0, step=10.0)
desv = st.sidebar.number_input("Desviación estándar", value=20.0, step=5.0)

# -----------------------------
# Generación de datos (cache)
# -----------------------------
@st.cache_data(show_spinner=False)
def generar_datos(seed: int, n_dias: int, n_categorias: int, mu: float, sigma: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fechas = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_dias, freq="D")
    categorias = [f"Cat_{i+1}" for i in range(n_categorias)]

    # Grilla de fechas x categorías
    df = pd.DataFrame([(f, c) for f in fechas for c in categorias], columns=["fecha", "categoria"])

    # Valores aleatorios (no negativos)
    df["valor"] = np.maximum(0, rng.normal(mu, sigma, size=len(df))).round(2)

    # Columnas útiles
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month

    # Día de semana en español SIN usar locales del sistema
    dias_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    df["dia_semana"] = df["fecha"].dt.weekday.map(lambda i: dias_es[int(i)])

    return df

df = generar_datos(seed, n_dias, n_categorias, media, desv)

# -----------------------------
# Resumen / Métricas
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filas", f"{len(df):,}".replace(",", "."))
col2.metric("Categorías", df["categoria"].nunique())
col3.metric("Rango de fechas", f"{df['fecha'].min().date()} → {df['fecha'].max().date()}")
col4.metric("Valor total", f"{df['valor'].sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

st.divider()

# -----------------------------
# Filtros
# -----------------------------
with st.expander("Filtros", expanded=False):
    opciones = sorted(df["categoria"].unique())
    cats_sel = st.multiselect("Filtrar categorías", options=opciones, default=opciones)
    # Controla el tipo de retorno de date_input (tupla para rango)
    fecha_ini, fecha_fin = st.date_input(
        "Rango de fechas",
        value=(df["fecha"].min().date(), df["fecha"].max().date())
    )

# Si el usuario desmarca todo, mantenemos todas las categorías para evitar DF vacío accidental
if not cats_sel:
    cats_sel = opciones

# Aplicar filtros (fecha_ini y fecha_fin ya son datetime.date)
df_filtrado = df[
    (df["categoria"].isin(cats_sel)) &
    (df["fecha"].dt.date >= fecha_ini) &
    (df["fecha"].dt.date <= fecha_fin)
].copy()

# -----------------------------
# Tabla descriptiva
# -----------------------------
st.subheader("Tabla descriptiva")
colA, colB = st.columns([2, 1])
with colA:
    st.dataframe(df_filtrado.head(50), use_container_width=True)
with colB:
    desc = df_filtrado["valor"].describe().to_frame(name="valor")
    st.dataframe(desc, use_container_width=True)

# -----------------------------
# Visualización 1: Línea (serie temporal agregada)
# -----------------------------
st.subheader("📈 Serie temporal (línea)")
if len(df_filtrado) > 0:
    serie = (
        df_filtrado.groupby("fecha", as_index=False)["valor"]
        .sum()
        .rename(columns={"valor": "valor_total"})
    )

    line_chart = (
        alt.Chart(serie)
        .mark_line(point=True)
        .encode(
            x=alt.X("fecha:T", title="Fecha"),
            y=alt.Y("valor_total:Q", title="Valor total"),
            tooltip=[alt.Tooltip("fecha:T", title="Fecha"), alt.Tooltip("valor_total:Q", title="Total")]
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(line_chart, use_container_width=True)
else:
    st.info("No hay datos para el rango/categorías seleccionados.")

# -----------------------------
# Visualización 2: Barras (suma por categoría)
# -----------------------------
st.subheader("📊 Suma por categoría (barras)")
if len(df_filtrado) > 0:
    barra_df = (
        df_filtrado.groupby("categoria", as_index=False)["valor"]
        .sum()
        .sort_values("valor", ascending=False)
    )

    bar_chart = (
        alt.Chart(barra_df)
        .mark_bar()
        .encode(
            x=alt.X("valor:Q", title="Suma de valor"),
            y=alt.Y("categoria:N", sort="-x", title="Categoría"),
            tooltip=[alt.Tooltip("categoria:N", title="Categoría"),
                     alt.Tooltip("valor:Q", title="Suma", format=",.2f")]
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(bar_chart, use_container_width=True)
else:
    st.info("No hay datos para mostrar en barras con los filtros actuales.")

# -----------------------------
# Descarga de datos
# -----------------------------
st.subheader("⬇️ Descargar datos")
csv = df_filtrado.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar CSV filtrado",
    data=csv,
    file_name="datos_filtrados.csv",
    mime="text/csv"
)

# -----------------------------
# Notas
# -----------------------------
with st.expander("Notas"):
    st.markdown(
        """
- Los datos se generan **aleatoriamente** en cada ejecución, controlados por la **semilla**.
- Puedes ajustar el número de días, categorías y la distribución (media/desviación) en el **sidebar**.
- No se usa `locale` del sistema; los días de la semana se traducen manualmente para evitar errores.
- La caché se invalida cuando cambias los parámetros, regenerando el dataset.
        """
    )
