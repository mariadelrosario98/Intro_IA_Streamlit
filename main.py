# App de Agricultura - EDA interactivo

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import re

# =========================
# Configuración de página
# =========================
st.set_page_config(page_title="Agricultura: EDA interactivo", page_icon="🌾", layout="wide")
st.title("🌾 Agricultura — EDA interactivo")
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
# Generación del dataset sintético
# =========================
# 10 columnas: fecha, finca_id, cultivo, region, lat, lon, area_ha, rendimiento_t_ha, lluvia_mm, ndvi
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
# Carga de CSV (opcional) — Reemplaza el dataset sintético
# =========================
st.subheader("📤 Cargar CSV (opcional)")
with st.expander("Usar mis propios datos (en lugar de los aleatorios)", expanded=False):
    fuente_cols = {
        "fecha": ["fecha", "date", "fecharegistro"],
        "finca_id": ["finca_id", "finca", "id_finca", "id"],
        "cultivo": ["cultivo", "crop", "variedad"],
        "region": ["region", "departamento", "zona"],
        "lat": ["lat", "latitude", "latitud"],
        "lon": ["lon", "lng", "long", "longitud", "longitude"],
        "area_ha": ["area_ha", "area", "hectareas", "ha"],
        "rendimiento_t_ha": ["rendimiento_t_ha", "rend_t_ha", "rendimiento", "yield_t_ha", "t_ha"],
        "lluvia_mm": ["lluvia_mm", "lluvia", "precipitacion_mm", "rain_mm"],
        "ndvi": ["ndvi", "indice_ndvi", "vigor"]
    }

    def _guess_col(cols, aliases):
        cols_low = [c.lower().strip() for c in cols]
        for a in aliases:
            if a in cols_low:
                return cols[cols_low.index(a)]
        return None

    def _auto_map_columns(df_csv: pd.DataFrame):
        mapping = {}
        for target, aliases in fuente_cols.items():
            mapped = _guess_col(df_csv.columns.tolist(), aliases)
            mapping[target] = mapped
        return mapping

    up = st.file_uploader("Sube un archivo CSV", type=["csv"])
    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        delimiter = st.selectbox("Delimitador", [",", ";", "\t", "|"], index=0)
    with col_up2:
        enc = st.selectbox("Codificación", ["utf-8", "latin-1", "utf-16"], index=0)

    if up is not None:
        try:
            df_csv = pd.read_csv(up, sep=delimiter, encoding=enc)
        except Exception as e:
            st.error(f"No pude leer el CSV: {e}")
            df_csv = None

        if df_csv is not None:
            st.caption(f"Archivo cargado: {up.name} · {len(df_csv):,} filas · {len(df_csv.columns)} columnas".replace(",", "."))
            st.dataframe(df_csv.head(10), use_container_width=True, height=220)

            # Intento de mapeo automático
            mapping = _auto_map_columns(df_csv)

            st.markdown("**Mapeo de columnas** (ajústalo si es necesario):")
            col_map1, col_map2, col_map3 = st.columns(3)
            required = list(fuente_cols.keys())

            # Controles de mapeo (en 3 columnas para no alargar)
            widgets = {}
            groups = [required[i::3] for i in range(3)]
            for ix, group in enumerate(groups):
                with [col_map1, col_map2, col_map3][ix]:
                    for tgt in group:
                        widgets[tgt] = st.selectbox(
                            f"{tgt}",
                            options=["—"] + df_csv.columns.tolist(),
                            index=(df_csv.columns.tolist().index(mapping[tgt]) + 1) if mapping[tgt] in df_csv.columns else 0
                        )

            if st.button("✅ Usar este CSV en el EDA"):
                sel = {k: v for k, v in widgets.items() if v != "—"}
                if len(sel) < len(required):
                    faltan = [k for k in required if k not in sel]
                    st.warning(f"Faltan columnas por mapear: {', '.join(faltan)}")
                else:
                    df_user = pd.DataFrame({
                        "fecha": df_csv[sel["fecha"]],
                        "finca_id": df_csv[sel["finca_id"]],
                        "cultivo": df_csv[sel["cultivo"]],
                        "region": df_csv[sel["region"]],
                        "lat": df_csv[sel["lat"]],
                        "lon": df_csv[sel["lon"]],
                        "area_ha": df_csv[sel["area_ha"]],
                        "rendimiento_t_ha": df_csv[sel["rendimiento_t_ha"]],
                        "lluvia_mm": df_csv[sel["lluvia_mm"]],
                        "ndvi": df_csv[sel["ndvi"]],
                    }).copy()

                    # Coerciones y limpieza ligera
                    df_user["fecha"] = pd.to_datetime(df_user["fecha"], errors="coerce")
                    for c in ["lat", "lon", "area_ha", "rendimiento_t_ha", "lluvia_mm", "ndvi"]:
                        df_user[c] = pd.to_numeric(df_user[c], errors="coerce")

                    before = len(df_user)
                    df_user = df_user.dropna(subset=["fecha", "lat", "lon"]).reset_index(drop=True)
                    dropped = before - len(df_user)

                    df_user["finca_id"] = df_user["finca_id"].astype(str)
                    df_user["cultivo"] = df_user["cultivo"].astype(str)
                    df_user["region"]  = df_user["region"].astype(str)

                    df_user = df_user[[
                        "fecha", "finca_id", "cultivo", "region", "lat", "lon",
                        "area_ha", "rendimiento_t_ha", "lluvia_mm", "ndvi"
                    ]].sort_values("fecha").reset_index(drop=True)

                    if dropped > 0:
                        st.info(f"Se descartaron {dropped} filas por fecha/lat/lon no válidas.")

                    # Reemplazar dataset base
                    df = df_user
                    st.success("✅ ¡Listo! Ahora el EDA usa **tu CSV**. Ajusta los filtros de arriba para explorar.")

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

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
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
    if len(df_f) > 0:
        chart_yield = (
            alt.Chart(df_f)
            .transform_bin(as_="bin_rend", field="rendimiento_t_ha", bin=alt.Bin(maxbins=30))
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
    else:
        st.info("No hay datos para el histograma con los filtros actuales.")

with right:
    st.markdown("**Lluvia promedio por región (mm)**")
    if len(df_f) > 0:
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
    else:
        st.info("No hay datos para el gráfico de lluvia.")

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
# Mapa (pydeck)
# =========================
st.subheader("🗺️ Mapa de fincas")
col_map1, col_map2, col_map3 = st.columns([1, 1, 2])
with col_map1:
    mostrar_mapa = st.checkbox("Mostrar mapa", value=True)
with col_map2:
    radio_pt = st.slider("Radio del punto (m)", min_value=500, max_value=4000, value=1500, step=100)
with col_map3:
    centrar_btn = st.button("📍 Centrar vista")

def _color_column(series: pd.Series) -> pd.Series:
    # paleta base en listas puras [r,g,b]
    palette = [
        [66, 135, 245], [245, 66, 93], [66, 245, 161], [166, 66, 245],
        [245, 171, 66], [66, 245, 236], [160, 160, 160], [100, 200, 100]
    ]
    keys = sorted(series.unique().tolist())
    mapping = {k: list(palette[i % len(palette)]) for i, k in enumerate(keys)}  # listas puras
    return series.map(mapping)

if mostrar_mapa and len(df_f) > 0:
    df_map = df_f.copy()
    if color_mapa == "cultivo":
        df_map["color"] = _color_column(df_map["cultivo"])
    else:
        df_map["color"] = _color_column(df_map["region"])

    # Estado de vista
    lat_c = float(df_map["lat"].mean())
    lon_c = float(df_map["lon"].mean())

    if centrar_btn:
        lat_c = float(df_map["lat"].mean())
        lon_c = float(df_map["lon"].mean())

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[lon, lat]',
        get_radius=radio_pt,
        get_fill_color='color',  # columna del DataFrame con [r,g,b]
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

# =========================
# =========================
# =========================
# 🤖 Agente de preguntas de agricultura (beta, mejorado)
# =========================
import re, unicodedata

st.divider()
st.subheader("🤖 Agente de preguntas agrícolas (beta)")

# --- Preferencias del agente
col_ag1, col_ag2 = st.columns([2, 3])
with col_ag1:
    usar_filtros = st.toggle(
        "Usar datos filtrados (df_f)", value=True,
        help="Si está activo, el agente responde con base en el subconjunto filtrado por tus controles de arriba."
    )
with col_ag2:
    modo_detalle = st.radio("Nivel de detalle de respuesta", ["Resumido", "Completo"], horizontal=True)

DATA_ACTUAL = df_f if usar_filtros else df
REGIONES = sorted(DATA_ACTUAL["region"].unique().tolist())
CULTIVOS = sorted(DATA_ACTUAL["cultivo"].unique().tolist())

# ---------- Helpers ----------
def _normalize(s: str) -> str:
    """Minúsculas + sin tildes + espacios compactos."""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return " ".join(s.split())

def _extract_entities(q: str):
    qn = _normalize(q)
    region = next((r for r in REGIONES if _normalize(r) in qn), None)
    cultivo = next((c for c in CULTIVOS if _normalize(c) in qn), None)
    return region, cultivo

def _subset(data: pd.DataFrame, region: str | None, cultivo: str | None) -> pd.DataFrame:
    d = data
    if region:
        d = d[d["region"] == region]
    if cultivo:
        d = d[d["cultivo"] == cultivo]
    return d

def _advice_block(d: pd.DataFrame) -> list[str]:
    """Heurísticas simples (ilustrativas) para recomendaciones."""
    tips = []
    if d.empty: return tips
    ndvi_m = float(d["ndvi"].mean())
    llv_m = float(d["lluvia_mm"].mean())
    rto_m = float(d["rendimiento_t_ha"].mean())
    if ndvi_m < 0.45:
        tips.append("NDVI bajo → revisar **estrés hídrico/nutricional**, malezas y plagas.")
    elif ndvi_m > 0.75:
        tips.append("NDVI alto → buen vigor; mantener **fitosanitario** y vigilar **exceso de humedad**.")
    if llv_m < 80:
        tips.append("Lluvia baja → considerar **riego suplementario** / conservación de humedad.")
    elif llv_m > 220:
        tips.append("Lluvia alta → reforzar **drenajes** y vigilar **enfermedades fungosas**.")
    if rto_m < 0.8 and "Café" in d["cultivo"].unique():
        tips.append("Café con bajo rendimiento → revisar **densidad, poda y nutrición N-P-K + Ca/Mg**.")
    if "Papa" in d["cultivo"].unique() and llv_m > 180:
        tips.append("Papa con alta lluvia → vigilar **tizón tardío** y mejorar **drenaje**.")
    return tips

def _describe_scope(region, cultivo, usar_filtros):
    sc = []
    sc.append("Fuente: " + ("**datos filtrados**" if usar_filtros else "**todos los datos**"))
    if region: sc.append(f"Región: **{region}**")
    if cultivo: sc.append(f"Cultivo: **{cultivo}**")
    return " · ".join(sc)

def _render_table(df_show: pd.DataFrame, cols: list[str], height=260):
    st.dataframe(df_show[cols], use_container_width=True, height=height)

def _trend_chart(d: pd.DataFrame, y_col: str, title: str):
    if d.empty:
        st.info("No hay datos para graficar."); return
    serie = d.groupby(["fecha"], as_index=False)[y_col].mean().rename(columns={y_col: "valor"})
    chart = (
        alt.Chart(serie).mark_line(point=True)
        .encode(x=alt.X("fecha:T", title="Fecha"),
                y=alt.Y("valor:Q", title=title),
                tooltip=[alt.Tooltip("fecha:T"), alt.Tooltip("valor:Q", format=",.2f")])
        .properties(height=280).interactive()
    )
    st.altair_chart(chart, use_container_width=True)

def _bar_chart(d: pd.DataFrame, group_col: str, y_col: str, title: str):
    if d.empty:
        st.info("No hay datos para graficar."); return
    g = d.groupby(group_col, as_index=False)[y_col].mean().rename(columns={y_col: "valor"})
    ch = (
        alt.Chart(g).mark_bar()
        .encode(x=alt.X("valor:Q", title=title),
                y=alt.Y(f"{group_col}:N", sort="-x", title=group_col.capitalize()),
                tooltip=[alt.Tooltip(f"{group_col}:N"), alt.Tooltip("valor:Q", format=",.2f")])
        .properties(height=320).interactive()
    )
    st.altair_chart(ch, use_container_width=True)

def _needs_cols(cols: list[str], data: pd.DataFrame) -> list[str]:
    return [c for c in cols if c not in data.columns]

# ---------- Respuestas de conocimiento (sin gráficos) ----------
def _knowledge_answer(q: str, region: str | None, cultivo: str | None, data: pd.DataFrame):
    qn = _normalize(q)
    # fertilizante / abono
    if re.search(r"\b(fertiliz|abono)\b", qn):
        crop = cultivo or "el cultivo"
        missing = _needs_cols(["fertilizante", "dosis_kg_ha"], data)
        nota_cols = ""
        if missing:
            nota_cols = f"\n\n> Para responder con datos propios necesitaríamos columnas como: {', '.join(missing)}."
        txt = (
            f"**No existe un ‘mejor fertilizante’ universal para {crop}.** "
            "La recomendación depende del **análisis de suelo** (pH, MO, P, K, Ca/Mg, CICE), edad del cultivo y objetivo de producción.\n\n"
            "En **café**, de forma general:\n"
            "- Se usa un esquema **NPK** con énfasis en **N** y **K**; el **P** se ajusta según suelo.\n"
            "- Fraccionar la aplicación en 3–4 eventos/año; complementar con **Ca/Mg** si el análisis lo indica.\n"
            "- Incorporar **materia orgánica** y manejo de **pH** (enmiendas) cuando aplique.\n\n"
            "👉 Paso práctico: realiza o consulta un análisis de suelo reciente y calcule la dosis objetivo con tablas locales (Cenicafé/extensionismo)."
            + nota_cols
        )
        return (txt, None, None)

    # plagas/enfermedades (IPM)
    if re.search(r"\b(roya|broca|plaga|enfermedad|mancha|hongo)\b", qn):
        crop = cultivo or "el cultivo"
        txt = (
            f"**Manejo integrado de plagas/enfermedades en {crop} (enfoque general):**\n"
            "- **Monitoreo** periódico (trampas/inspección) y umbrales de intervención.\n"
            "- **Culturales**: podas sanitarias, manejo de sombra, ventilación, nutrición balanceada.\n"
            "- **Biológico**: uso de controladores biológicos donde aplique.\n"
            "- **Químico**: solo si supera umbral; rotar modos de acción y respetar periodos de carencia.\n"
            "Si tienes fechas de incidencia/parcelas, puedo graficar tendencias por lote."
        )
        return (txt, None, None)

    # riego / agua
    if re.search(r"\b(riego|lamina|evapotranspiracion|et0)\b", qn):
        txt = (
            "**Riego (orientación general):** calcule la lámina ≈ ETc = ET₀ × Kc, "
            "ajuste por eficiencia del sistema y fraccione según textura/salinidad. "
            "Con columnas como `et0_mm`, `kc`, `riego_mm` puedo estimar déficits y sugerir ventanas de riego."
        )
        return (txt, None, None)

    # suelos / pH / MO
    if re.search(r"\b(suelo|ph|materia organica|m o|m\.o\.)\b", qn):
        txt = (
            "**Suelos:** mantener pH objetivo del cultivo (en café ~5.2–5.8, referencia general), "
            "aplicar enmiendas (cal/dolomita/yeso) según saturación de bases y Al. "
            "La **MO** mejora estructura y CICE; medirla anual/bianual. "
            "Si cargas pH, MO, Ca, Mg, K, P puedo armar balances y recomendaciones más finas."
        )
        return (txt, None, None)

    # densidad/siembra
    if re.search(r"\b(densidad|siembra|espaciamiento)\b", qn):
        txt = (
            "**Densidad/siembra:** depende de variedad, pendiente, mecanización y régimen hídrico. "
            "Busca equilibrio entre **interceptación de luz** y **ventilación**. "
            "Con columnas de `marco_x`, `marco_y` o `plantas_ha` puedo contrastar contra rendimiento/NDVI."
        )
        return (txt, None, None)

    return None  # no matchea intención de conocimiento

# ---------- Router principal ----------
def answer_question(q: str, data: pd.DataFrame):
    """Devuelve (texto_respuesta, df_opcional, extra) donde:
       - extra == 'tabla' para tablas
       - extra == ('barra', group_col, y_col) para barras
       - extra == ('tendencia', y_col) para series temporales
    """
    ql = q.lower()
    qn = _normalize(q)
    region, cultivo = _extract_entities(q)
    d = _subset(data, region, cultivo)

    # 1) Intentos de conocimiento (antes de comparativos)
    k = _knowledge_answer(q, region, cultivo, data)
    if k is not None:
        return k  # texto sin gráficos

    # 2) Si pide Top/Mejores → tabla
    if ("top" in qn) or ("mejores" in qn):
        n = 10
        m = re.search(r"top\s*(\d+)", qn) or re.search(r"top(\d+)", qn)
        if m:
            try: n = max(1, min(100, int(m.group(1))))
            except: pass
        if d.empty: return ("No hay datos para esa selección.", None, None)
        top = d.sort_values("rendimiento_t_ha", ascending=False).head(n)
        text = f"Top {len(top)} fincas por **rendimiento (t/ha)** · {_describe_scope(region, cultivo, usar_filtros)}"
        return (text, top[["finca_id","region","cultivo","rendimiento_t_ha","area_ha","ndvi","lluvia_mm"]], "tabla")

    if d.empty:
        return ("No encontré datos que coincidan con tu consulta. Ajusta filtros/región/cultivo e inténtalo de nuevo.", None, None)

    # 3) Rendimiento / Lluvia / NDVI
    if ("rend" in qn) or ("productividad" in qn):
        text = f"Resumen de **rendimiento (t/ha)** · {_describe_scope(region, cultivo, usar_filtros)}"
        if ("por region" in qn) or ("por region" in qn):
            return (text, d, ("barra","region","rendimiento_t_ha"))
        if ("por cultivo" in qn):
            return (text, d, ("barra","cultivo","rendimiento_t_ha"))
        if ("tendencia" in qn) or ("serie" in qn):
            return (text, d, ("tendencia","rendimiento_t_ha"))
        s = d["rendimiento_t_ha"].describe()[["count","mean","std","min","max"]]
        text += f"\n- n={int(s['count'])} · media={s['mean']:.2f} · σ={s['std']:.2f} · min={s['min']:.2f} · max={s['max']:.2f}"
        return (text, None, None)

    if ("lluvia" in qn) or ("precipitaci" in qn):
        text = f"**Lluvia (mm)** · {_describe_scope(region, cultivo, usar_filtros)}"
        if ("por region" in qn):
            return (text, d, ("barra","region","lluvia_mm"))
        if ("por cultivo" in qn):
            return (text, d, ("barra","cultivo","lluvia_mm"))
        if ("tendencia" in qn) or ("serie" in qn):
            return (text, d, ("tendencia","lluvia_mm"))
        s = d["lluvia_mm"].describe()[["count","mean","std","min","max"]]
        text += f"\n- n={int(s['count'])} · media={s['mean']:.1f} · σ={s['std']:.1f} · min={s['min']:.1f} · max={s['max']:.1f}"
        return (text, None, None)

    if ("ndvi" in qn) or ("vigor" in qn):
        text = f"**NDVI** · {_describe_scope(region, cultivo, usar_filtros)}"
        if ("por region" in qn):
            return (text, d, ("barra","region","ndvi"))
        if ("por cultivo" in qn):
            return (text, d, ("barra","cultivo","ndvi"))
        if ("tendencia" in qn) or ("serie" in qn):
            return (text, d, ("tendencia","ndvi"))
        s = d["ndvi"].describe()[["count","mean","std","min","max"]]
        text += f"\n- n={int(s['count'])} · media={s['mean']:.3f} · σ={s['std']:.3f} · min={s['min']:.3f} · max={s['max']:.3f}"
        return (text, None, None)

    # 4) Comparativos rápidos (solo si no es conocimiento)
    if ("?" in q) or ("cual" in qn) or ("cual es" in qn) or ("mejor" in qn):
        if region and not cultivo:
            text = f"Comparativo de **rendimiento** por cultivo en **{region}**"
            return (text, d, ("barra","cultivo","rendimiento_t_ha"))
        if cultivo and not region:
            text = f"Comparativo de **rendimiento** por región para **{cultivo}**"
            return (text, d, ("barra","region","rendimiento_t_ha"))

    # 5) Fallback con señales
    base = f"Esto es lo que puedo decir con los datos · {_describe_scope(region, cultivo, usar_filtros)}"
    tips = _advice_block(d)
    if modo_detalle == "Completo":
        base += f"\n- Observaciones: {len(d)}"
        base += f"\n- Rend. medio: {d['rendimiento_t_ha'].mean():.2f} t/ha · NDVI medio: {d['ndvi'].mean():.3f} · Lluvia media: {d['lluvia_mm'].mean():.1f} mm"
    if tips:
        base += "\n\nSugerencias:\n- " + "\n- ".join(tips)
    return (base, None, None)

# --- Historial de chat
if "agro_chat" not in st.session_state:
    st.session_state.agro_chat = []

for role, content in st.session_state.agro_chat:
    with st.chat_message(role):
        st.markdown(content)

# --- Entrada del usuario
prompt = st.chat_input("Haz una pregunta (p. ej., 'Mejor fertilizante para café', 'Top 5 por rendimiento en Antioquia', 'Tendencia de NDVI para Maíz').")
if prompt:
    st.session_state.agro_chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    texto, payload, extra = answer_question(prompt, DATA_ACTUAL)

    with st.chat_message("assistant"):
        st.markdown(texto)
        if extra is None:
            if isinstance(payload, pd.DataFrame):
                _render_table(payload, ["finca_id","region","cultivo","rendimiento_t_ha","area_ha","ndvi","lluvia_mm"])
        elif isinstance(extra, str) and extra == "tabla":
            if isinstance(payload, pd.DataFrame) and not payload.empty:
                _render_table(payload, ["finca_id","region","cultivo","rendimiento_t_ha","area_ha","ndvi","lluvia_mm"])
            else:
                st.info("No hay filas para mostrar.")
        elif isinstance(extra, tuple):
            if extra[0] == "barra":
                _, group_col, y_col = extra
                _bar_chart(payload, group_col, y_col, title=y_col.replace("_", " ").upper())
            elif extra[0] == "tendencia":
                _, y_col = extra
                _trend_chart(payload, y_col, title=y_col.replace("_", " ").upper())

    st.session_state.agro_chat.append(("assistant", texto))
