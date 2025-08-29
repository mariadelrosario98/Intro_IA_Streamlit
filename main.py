# =========================
# 🤖 Agente de preguntas de agricultura (beta)
# =========================
import re

st.divider()
st.subheader("🤖 Agente de preguntas agrícolas (beta)")

# --- Preferencias del agente
col_ag1, col_ag2 = st.columns([2, 3])
with col_ag1:
    usar_filtros = st.toggle("Usar datos filtrados (df_f)", value=True,
                             help="Si está activo, el agente responde con base en el subconjunto filtrado por tus controles de arriba.")
with col_ag2:
    modo_detalle = st.radio("Nivel de detalle de respuesta", ["Resumido", "Completo"], horizontal=True)

DATA_ACTUAL = df_f if usar_filtros else df

REGIONES = sorted(df["region"].unique().tolist())
CULTIVOS = sorted(df["cultivo"].unique().tolist())

def _extract_entities(q: str):
    ql = q.lower()
    region = next((r for r in REGIONES if r.lower() in ql), None)
    cultivo = next((c for c in CULTIVOS if c.lower() in ql), None)
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
    if d.empty:
        return tips

    ndvi_m = float(d["ndvi"].mean())
    llv_m = float(d["lluvia_mm"].mean())
    rto_m = float(d["rendimiento_t_ha"].mean())

    # Señales básicas (ajusta umbrales si lo deseas)
    if ndvi_m < 0.45:
        tips.append("NDVI bajo: revisar **estrés hídrico/nutricional** y cobertura vegetal; evaluar malezas o plagas.")
    elif ndvi_m > 0.75:
        tips.append("NDVI alto: buen vigor; mantener **manejo fitosanitario** y monitoreo de **exceso de humedad**.")
    if llv_m < 80:
        tips.append("Lluvia baja: considerar **riego suplementario** o conservación de humedad (mulch, coberturas).")
    elif llv_m > 220:
        tips.append("Lluvia elevada: reforzar **drenajes**, vigilar **enfermedades fungosas** y **lixiviación** de nutrientes.")
    if rto_m < 0.8 and "Café" in d["cultivo"].unique():
        tips.append("Rendimiento cafetalero bajo: revisar **densidad de siembra**, poda y **nutrición N-P-K + Ca/Mg**.")
    if "Papa" in d["cultivo"].unique() and llv_m > 180:
        tips.append("Papa con alta lluvia: vigilar **tizón tardío**; fortalecer **cobertura y drenaje**.")

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
        st.info("No hay datos para graficar.")
        return
    serie = (
        d.groupby(["fecha"], as_index=False)[y_col]
        .mean()
        .rename(columns={y_col: "valor"})
    )
    chart = (
        alt.Chart(serie)
        .mark_line(point=True)
        .encode(
            x=alt.X("fecha:T", title="Fecha"),
            y=alt.Y("valor:Q", title=title),
            tooltip=[alt.Tooltip("fecha:T"), alt.Tooltip("valor:Q", format=",.2f")]
        )
        .properties(height=280)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

def _bar_chart(d: pd.DataFrame, group_col: str, y_col: str, title: str):
    if d.empty:
        st.info("No hay datos para graficar.")
        return
    g = d.groupby(group_col, as_index=False)[y_col].mean().rename(columns={y_col: "valor"})
    ch = (
        alt.Chart(g)
        .mark_bar()
        .encode(
            x=alt.X("valor:Q", title=title),
            y=alt.Y(f"{group_col}:N", sort="-x", title=group_col.capitalize()),
            tooltip=[alt.Tooltip(f"{group_col}:N"), alt.Tooltip("valor:Q", format=",.2f")]
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(ch, use_container_width=True)

def answer_question(q: str, data: pd.DataFrame):
    """Devuelve (texto_respuesta, df_opcional, tipo_extra) donde tipo_extra puede ser 'tabla', 'barra', 'tendencia'."""
    region, cultivo = _extract_entities(q)
    d = _subset(data, region, cultivo)

    if d.empty:
        return ("No encontré datos que coincidan con tu consulta. Ajusta filtros/región/cultivo e inténtalo de nuevo.", None, None)

    ql = q.lower()

    # --- Top N por rendimiento
    if "top" in ql or "mejores" in ql:
        n = 10
        m = re.search(r"top\s*(\d+)", ql)
        if m:
            try: n = max(1, min(100, int(m.group(1))))
            except: pass
        top = d.sort_values("rendimiento_t_ha", ascending=False).head(n)
        text = f"Top {len(top)} fincas por **rendimiento (t/ha)** · {_describe_scope(region, cultivo, usar_filtros)}"
        return (text, top[["finca_id","region","cultivo","rendimiento_t_ha","area_ha","ndvi","lluvia_mm"]], "tabla")

    # --- Estadística de rendimiento
    if "rend" in ql or "productividad" in ql:
        text = f"Resumen de **rendimiento (t/ha)** · {_describe_scope(region, cultivo, usar_filtros)}"
        if "por región" in ql:
            return (text, d, ("barra","region","rendimiento_t_ha"))
        if "por cultivo" in ql:
            return (text, d, ("barra","cultivo","rendimiento_t_ha"))
        if "tendencia" in ql or "serie" in ql:
            return (text, d, ("tendencia","rendimiento_t_ha"))
        # default resumido
        stats = d["rendimiento_t_ha"].describe()[["count","mean","std","min","max"]]
        text += f"\n- n={int(stats['count'])} · media={stats['mean']:.2f} · σ={stats['std']:.2f} · min={stats['min']:.2f} · max={stats['max']:.2f}"
        return (text, None, None)

    # --- Lluvia
    if "lluvia" in ql or "precipitaci" in ql:
        text = f"**Lluvia (mm)** · {_describe_scope(region, cultivo, usar_filtros)}"
        if "por región" in ql:
            return (text, d, ("barra","region","lluvia_mm"))
        if "por cultivo" in ql:
            return (text, d, ("barra","cultivo","lluvia_mm"))
        if "tendencia" in ql or "serie" in ql:
            return (text, d, ("tendencia","lluvia_mm"))
        stats = d["lluvia_mm"].describe()[["count","mean","std","min","max"]]
        text += f"\n- n={int(stats['count'])} · media={stats['mean']:.1f} · σ={stats['std']:.1f} · min={stats['min']:.1f} · max={stats['max']:.1f}"
        return (text, None, None)

    # --- NDVI
    if "ndvi" in ql or "vigor" in ql:
        text = f"**NDVI** · {_describe_scope(region, cultivo, usar_filtros)}"
        if "por región" in ql:
            return (text, d, ("barra","region","ndvi"))
        if "por cultivo" in ql:
            return (text, d, ("barra","cultivo","ndvi"))
        if "tendencia" in ql or "serie" in ql:
            return (text, d, ("tendencia","ndvi"))
        stats = d["ndvi"].describe()[["count","mean","std","min","max"]]
        text += f"\n- n={int(stats['count'])} · media={stats['mean']:.3f} · σ={stats['std']:.3f} · min={stats['min']:.3f} · max={stats['max']:.3f}"
        return (text, None, None)

    # --- Comparativos rápidos
    if "¿" in q or "?" in q or "cual" in ql or "cuál" in ql or "mejor" in ql:
        # Ej: "¿Cuál cultivo rinde más en Antioquia?"
        # Si hay región -> comparar por cultivo; si hay cultivo -> comparar por región.
        if region and not cultivo:
            text = f"Comparativo de **rendimiento** por cultivo en **{region}**"
            return (text, d, ("barra","cultivo","rendimiento_t_ha"))
        if cultivo and not region:
            text = f"Comparativo de **rendimiento** por región para **{cultivo}**"
            return (text, d, ("barra","region","rendimiento_t_ha"))

    # --- Recomendaciones básicas
    if "recom" in ql or "suger" in ql or "consejo" in ql:
        tips = _advice_block(d)
        if tips:
            txt = f"Recomendaciones (heurísticas) · {_describe_scope(region, cultivo, usar_filtros)}\n- " + "\n- ".join(tips)
        else:
            txt = "Sin señales claras para recomendar con esta muestra."
        return (txt, None, None)

    # --- Fallback
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
prompt = st.chat_input("Haz una pregunta (ej.: 'Top 5 por rendimiento en Antioquia', 'Tendencia de NDVI para Café', 'lluvia por región', 'recomendaciones en Tolima').")
if prompt:
    st.session_state.agro_chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    texto, payload, extra = answer_question(prompt, DATA_ACTUAL)

    with st.chat_message("assistant"):
        st.markdown(texto)
        if extra is None and isinstance(payload, pd.DataFrame):
            _render_table(payload, ["finca_id","region","cultivo","rendimiento_t_ha","area_ha","ndvi","lluvia_mm"])
        elif isinstance(extra, tuple):
            # extra puede ser ("barra", group_col, y_col) o ("tendencia", y_col)
            if extra[0] == "barra":
                _, group_col, y_col = extra
                _bar_chart(payload, group_col, y_col, title=y_col.replace("_", " ").upper())
            elif extra[0] == "tendencia":
                _, y_col = extra
                _trend_chart(payload, y_col, title=y_col.replace("_", " ").upper())

    # Guardar la respuesta en el historial
    st.session_state.agro_chat.append(("assistant", texto))
