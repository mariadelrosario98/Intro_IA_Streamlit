# App de Agricultura - EDA interactivo con datos aleatorios (corregida)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

# =========================
# Configuración de página
# =========================
st.set_page_config(page_title="Agricultura: EDA interactivo", page_icon="🌾", layout="wide")
st.title("🌾 Agricultura — EDA interactivo con datos aleatorios")
st.caption("500 observaciones • 10 columnas • Controles interactivos, gráficos y mapa")

# =========================
# Sidebar - Controles globales
# =========================
st.sidebar.header("Parámetros de los datos")
seed = st.sidebar.number_input("Semilla aleatoria", min_value=0, max_value=10_000, value=123, step=1)
n_obs = 500  # según requerimiento
st.sidebar.write(f"Observaciones: **{n_obs}** (fijas)")
st.sidebar.divider()

# =========================
# Generación del dataset
# 10 columnas:
# fecha, finca_id, cultivo, region, lat, lon, area_ha, rendimiento_t_ha, lluvia_mm, ndvi
# =========================
@st.cache_data(show_spinner=False)
def generar_datos_agro(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Fechas dentro del último año
    fechas = pd.to_datetime("today").normalize() - pd.to_timedelta(rng.integers(0, 365, size=n), unit="D")

    # Fincas y regiones
    fincas = [f"F{str(i).zfill(4)}" for i in rng.integers(1, 3000, size=n)]
    cultivos = rng.choice(
        ["Café", "Maíz", "Arroz", "Cacao", "Banano", "Papa", "Soja"],
        size=n,
        p=[0.22, 0.18, 0.14, 0.10, 0.14, 0.12, 0.10]
    )
    regiones = rng.choice(
        ["Antioquia", "Huila", "Tolima", "Cundinamarca", "Santander", "Cesar"],
        size=n
    )

    # Lat/Lon aproximados de Colombia (ruido alrededor de centroides regionales)
    centros = {
        "Antioquia": (6.25, -75.56),
        "Huila": (2.94, -75.28),
        "Tolima": (4.44, -75.24),
        "Cundinamarca": (4.71, -74.07),
        "Santander": (7.13, -73.13),
        "Cesar": (10.46, -73.25),
    }
    lat = np.array([centros[r][0] for r in regiones]) + rng.normal(0, 0.35, size=n)
    lon = np.array([centros[r][1] for r in regiones]) + rng.normal(0, 0.35, size=n)

    # Área (ha), rendimiento (t/ha), lluvia (mm), NDVI
    area_ha = np.clip(rng.normal(5, 3, size=n), 0.2, 50).round(2)
    base_yield = {
        "Café": 1.2, "Maíz": 5.0, "Arroz": 4.5, "Cacao": 0.9,
        "Banano": 30.0, "Papa": 16.0, "Soja": 2.8
    }
    lluvia_mm = np.clip(rng.normal(120, 60, size=n), 0, 400).round(1)
    ruido_y = rng.normal(0, 0.15, size=n)
    rendimiento_t_ha = (
        np.array([base_yield[c] for c in cultivos]) * (1 + (lluvia_mm - 120)/800) * (1 + ruido_y)
    )
    rendimiento_t_ha = np.clip(rendimiento_t_ha, 0.2, None).round(2)

    ndvi = np.clip(rng.normal(0.65, 0.12, size=n), 0.1, 0.95).round(3)

    df = pd.DataFrame({
        "fecha": fechas,
        "finca_id": fincas,
        "cultivo": cultivos,
        "region": regiones,
        "lat": lat.astype(float).round(5),
        "lon": lon.astype(float).round(5),
        "area_ha": area_ha.astype(float),
        "rendimiento_t_ha": rendimiento_t_ha.astype(float),
        "lluvia_mm": lluvia_mm.astype(float),
        "ndvi": ndvi.astype(float)
    }).sort_values("fecha").reset_index(drop=True)

    return df

df = generar_datos_agro(seed, n_obs)

# =========================
# Filtros interactivos
# =========================
with st.expander("🧰 Filtros y opciones", expanded=True):
    cols = st.columns(3)
    with cols[0]:
        regiones_sel = st.multiselect(
            "Regiones",
            options=sorted(df["region"].unique()),
            default=sorted(df["region"].unique())
        )
    with cols[1]:
        cultivos_sel = st.multiselect(
            "Cultivos",
            options=sorted(df["cultivo"].unique()),
            default=sorted(df["cultivo"].unique())
        )
    with cols[2]:
        fecha_rango = st.slider(
            "Rango de fechas",
            min_value=df["fecha"].min().date(),
            max_value=df["fecha"].max().date(),
            value=(df["fecha"].min().date(), df["fecha"].max().date())
        )

    colx, coly, colz = st.columns(3)
    with colx:
        area_rango = st.slider(
            "Área (ha)",
            min_value=float(df["area_ha"].min()),
            max_value=float(df["area_ha"].max()),
            value=(float(df["area_ha"].min()), float(df["area_ha"].max()))
        )
    with coly:
        yield_rango = st.slider(
            "Rendimiento (t/ha)",
            min_value=float(df["rendimiento_t_ha"].min()),
            max_value=float(df["rendimiento_t_ha"].max()),
            value=(float(df["rendimiento_t_ha"].min()), float(df["rendimiento_t_ha"].max()))
        )
    with colz:
        ndvi_min = st.slider(
            "NDVI mínimo",
            min_value=float(df["ndvi"].min()),
            max_value=float(df["ndvi"].max()),
            value=float(df["ndvi"].min())
        )

    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        mostrar_datos = st.checkbox("Mostrar tabla", value=True)
    with c2:
        boton_recalcular = st.button("🔄 Regenerar datos")
    with c3:
        marcar_top = st.checkbox("Marcar top 10 por rendimiento", value=False)
    with c4:
        color_mapa = st.radio("Color del mapa por:", ["cultivo", "region"], horizontal=True)

# Recalcular (cambia semilla para forzar nuevo set)
if boton_recalcular:
    seed = int(seed) + 1
    df = generar_datos_agro(seed, n_obs)

# Aplicar filtros
mask = (
    df["region"].isin(regiones_sel) &
    df["cultivo"].isin(cultivos_sel) &
    (df["fecha"].dt.date >= fecha_rango[0]) &
    (df["fecha"].dt.date <= fecha_rango[1]) &
    (df["area_ha"].between(area_rango[0], area_rango[1])) &
    (df["rendimiento_t_ha"].between(yield_rango[0], yield_rango[1])) &
    (df["ndvi"] >= ndvi_min)
)
df_f = df[mask].copy()

# Top 10 por rendimiento (flag)
df_f["is_top"] = False
if marcar_top and len(df_f) > 0:
    top_idx = df_f["rendimiento_t_ha"].nlargest(10).index
    df_f.loc[top_idx, "is_top"] = True

# =========================
# Métricas
# =========================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Muestras", f"{len(df_f):,}".replace(",", "."))
m2.metric("Área total (ha)", f"{df_f['area_ha'].sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
m3.metric("Rend. medio (t/ha)", f"{df_f['rendimiento_t_ha'].mean():.2f}".replace(".", ","))
m4.metric("NDVI medio", f"{df_f['ndvi'].mean():.3f}".replace(".", ","))

st.divider()

# =========================
# Tabla
# =========================
if mostrar_datos:
    st.subheader("📋 Datos filtrados")
    st.dataframe(df_f, use_container_width=True, height=260)

# =========================
# Gráficos
# =========================
st.subheader("📈 Visualizaciones")

left, right = st.columns(2, gap="large")

with left:
    st.markdown("**Distribución de rendimiento (t/ha)**")
    chart_yield = (
        alt.Chart(df_f)
        .transform_bin("bin_rend", field="rendimiento_t_ha", bin=alt.Bin(maxbins=30))
        .mark_bar()
        .encode(
            x=alt.X("bin_rend:Q", title="Rendimiento (t/ha)"),
            y=alt.Y("count():Q", title="Frecuencia"),
            tooltip=[alt.Tooltip("count():Q", title="Frecuencia")]
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart_yield, use_container_width=True)

with right:
    st.markdown("**Lluvia promedio por región (mm)**")
    lluvia_region = (
        df_f.groupby("region", as_index=False)["lluvia_mm"].mean()
        .rename(columns={"lluvia_mm": "lluvia_prom"})
    )
    chart_lluvia = (
        alt.Chart(lluvia_region)
        .mark_bar()
        .encode(
            x=alt.X("lluvia_prom:Q", title="Lluvia promedio (mm)"),
            y=alt.Y("region:N", sort="-x", title="Región"),
            tooltip=[alt.Tooltip("region:N"), alt.Tooltip("lluvia_prom:Q", format=",.1f")]
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart_lluvia, use_container_width=True)

st.markdown("**Rendimiento medio por cultivo (línea temporal)**")
if len(df_f) > 0:
    serie = (
        df_f.groupby(["fecha", "cultivo"], as_index=False)["rendimiento_t_ha"]
        .mean()
        .rename(columns={"rendimiento_t_ha": "rend_medio"})
    )
    chart_line = (
        alt.Chart(serie)
        .mark_line(point=True)
        .encode(
            x=alt.X("fecha:T", title="Fecha"),
            y=alt.Y("rend_medio:Q", title="Rendimiento medio (t/ha)"),
            color="cultivo:N",
            tooltip=[alt.Tooltip("fecha:T"), "cultivo:N", alt.Tooltip("rend_medio:Q", format=",.2f")]
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(chart_line, use_container_width=True)
else:
    st.info("No hay datos para la serie temporal con los filtros actuales.")

st.divider()

# =========================
# Mapa (pydeck) — CORREGIDO
# =========================
st.subheader("🗺️ Mapa de fincas")
col_map1, col_map2, col_map3 = st.columns([1, 1, 2])
with col_map1:
    mostrar_mapa = st.checkbox("Mostrar mapa", value=True)
with col_map2:
    radio_pt = st.slider("Radio del punto (m)", min_value=500, max_value=4000, value=1500, step=100)
with col_map3:
    centrar_btn = st.button("📍 Centrar vista")

def _color_column(series):
    # paleta base en listas puras [r,g,b]
    palette = [
        [66, 135, 245], [245, 66, 93], [66, 245, 161], [166, 66, 245],
        [245, 171, 66], [66, 245, 236], [160, 160, 160], [100, 200, 100]
    ]
    keys = sorted(series.unique().tolist())
    mapping = {k: list(palette[i % len(palette)]) for i, k in enumerate(keys)}  # <- listas puras
    return series.map(mapping)

if mostrar_mapa and len(df_f) > 0:
    # Columna 'color' por fila (listas puras)
    df_f = df_f.copy()
    if color_mapa == "cultivo":
        df_f["color"] = _color_column(df_f["cultivo"])
    else:
        df_f["color"] = _color_column(df_f["region"])

    # Estado de vista
    lat_c = float(df_f["lat"].mean())
    lon_c = float(df_f["lon"].mean())

    if centrar_btn:
        # Recalcula centro según filtros actuales
        lat_c = float(df_f["lat"].mean())
        lon_c = float(df_f["lon"].mean())

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_f,
        get_position='[lon, lat]',
        get_radius=radio_pt,
        get_fill_color='color',   # <- columna del DataFrame
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=lat_c,
        longitude=lon_c,
        zoom=5,
        pitch=0
    )

    tooltip = {
        "html": "<b>Finca:</b> {finca_id}<br/>"
                "<b>Cultivo:</b> {cultivo}<br/>"
                "<b>Región:</b> {region}<br/>"
                "<b>Rendimiento:</b> {rendimiento_t_ha} t/ha<br/>"
                "<b>NDVI:</b> {ndvi}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
elif mostrar_mapa:
    st.info("No hay puntos para mostrar con los filtros actuales.")

st.divider()

# =========================
# Descargas y acciones
# =========================
st.subheader("⬇️ Exportar / Acciones")
cA, cB, cC = st.columns([2, 2, 3])
with cA:
    csv = df_f.drop(columns=["is_top"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV filtrado", data=csv, file_name="agro_filtrado.csv", mime="text/csv")
with cB:
    if st.button("📌 Marcar top 10 y mostrar solo esos"):
        tmp = df_f.sort_values("rendimiento_t_ha", ascending=False).head(10)
        st.dataframe(tmp, use_container_width=True)
with cC:
    st.success("Tip: Ajusta el **radio del punto** y usa **Centrar vista** para explorar mejor el mapa.")
