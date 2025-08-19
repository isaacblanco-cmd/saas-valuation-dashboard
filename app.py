
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="SaaS Valuation Dashboard", page_icon="📊", layout="wide")

# ---------------------------
# Config
# ---------------------------
DEFAULT_MULTIPLES = {
    "SaaS Infra / DevTools": 10.0,
    "FinTech": 12.0,
    "HealthTech": 8.0,
    "Horizontal B2B SaaS": 8.0,
    "Vertical SaaS": 6.0,
    "EdTech": 6.0,
    "E-commerce Enablement": 5.0,
    "Otro": 7.0,
}

def fmt_eur(x: float) -> str:
    try:
        return f"{x:,.0f} €".replace(",", ".")
    except Exception:
        return str(x)

@st.cache_data
def read_any(file):
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, engine="openpyxl")
    return df

def ensure_datetime(s: pd.Series):
    out = pd.to_datetime(s, errors="coerce")
    # If dates without day, pandas may set 1900. Try to coerce year-month strings too.
    return out

def project_monthly_arr(arr0: float, growth_m: float, churn_m: float, months: int):
    """Simple compounding using net growth (growth - churn)."""
    net = (growth_m - churn_m)
    series = []
    current = float(arr0)
    for _ in range(months):
        current = current * (1.0 + net)
        series.append(current)
    return pd.Series(series)

def build_projection_df(arr0: float, growth_m: float, churn_m: float, months: int, start_date: pd.Timestamp | None):
    base = project_monthly_arr(arr0, growth_m, churn_m, months)
    optimistic = project_monthly_arr(arr0, growth_m * 1.20, churn_m * 0.75, months)
    pessimistic = project_monthly_arr(arr0, max(growth_m * 0.80, 0.0), churn_m * 1.25, months)
    df = pd.DataFrame({
        "Base": base,
        "Optimista": optimistic,
        "Pesimista": pessimistic
    })
    if start_date is not None:
        idx = pd.date_range(start=start_date + pd.offsets.MonthBegin(1), periods=months, freq="MS")
        df.index = idx
    return df

def valuation_at_years(arr0: float, growth_m: float, churn_m: float, years: int, multiple: float) -> float:
    months = years * 12
    arr_n = arr0 * ((1.0 + (growth_m - churn_m)) ** months)
    return arr_n * multiple

# ---------------------------
# UI
# ---------------------------
st.title("📊 Valuación de Startup SaaS (conectado a Excel)")

with st.sidebar:
    st.header("Datos de entrada")
    uploaded = st.file_uploader("Sube tu Excel/CSV (histórico 2023-2025)", type=["xlsx", "xls", "csv"])
    st.caption("El archivo puede contener columnas como: Fecha, ARR, MRR, Crecimiento_mensual, Churn_mensual. Si no existen, las estimamos.")

df_hist = None
arr_current = 1_000_000.0
growth_m_default = 0.05
churn_m_default = 0.02
last_date = None

if uploaded is not None:
    try:
        df0 = read_any(uploaded)
        if df0.empty:
            st.warning("El archivo está vacío.")
        else:
            st.subheader("Vista previa de datos")
            st.dataframe(df0.head(10), use_container_width=True)

            st.markdown("### Mapeo de columnas")
            c1, c2, c3, c4 = st.columns(4)
            date_col = c1.selectbox("Columna de fecha", options=list(df0.columns))
            arr_or_mrr_col = c2.selectbox("Columna ARR o MRR", options=list(df0.columns))
            kind = c3.radio("¿Es ARR o MRR?", options=["ARR", "MRR"], horizontal=True)
            growth_col = c4.selectbox("Crecimiento mensual (%) [opcional]", options=["(ninguna)"] + list(df0.columns))
            churn_col = st.selectbox("Churn mensual (%) [opcional]", options=["(ninguna)"] + list(df0.columns))

            df = df0.copy()
            df[date_col] = ensure_datetime(df[date_col])
            df = df.sort_values(date_col)
            df = df.dropna(subset=[date_col])

            # ARR derivado
            arr_series = pd.to_numeric(df[arr_or_mrr_col], errors="coerce")
            if kind == "MRR":
                arr_series = arr_series * 12.0
            df["ARR"] = arr_series

            # growth / churn
            if growth_col != "(ninguna)":
                df["growth_m"] = pd.to_numeric(df[growth_col], errors="coerce") / 100.0
            else:
                df["growth_m"] = df["ARR"].pct_change().clip(lower=-1.0).fillna(0.0)
            if churn_col != "(ninguna)":
                df["churn_m"] = pd.to_numeric(df[churn_col], errors="coerce") / 100.0
            else:
                df["churn_m"] = 0.0

            df = df.dropna(subset=["ARR"])
            if df.empty:
                st.error("No se pudieron obtener valores válidos de ARR.")
            else:
                last_row = df.iloc[-1]
                last_date = pd.to_datetime(last_row[date_col])
                arr_current = float(last_row["ARR"])
                growth_m_default = float(last_row["growth_m"])
                churn_m_default = float(last_row["churn_m"])
                df_hist = df[[date_col, "ARR"]].rename(columns={date_col: "Fecha"})
    except Exception as e:
        st.error(f"Error leyendo archivo: {e}")

st.markdown("### Parámetros de escenario")
c1, c2, c3 = st.columns(3)
arr_input = c1.number_input("ARR actual (€)", value=float(arr_current), step=50_000.0, format="%.0f")
growth_input = c2.slider("Crecimiento mensual (%)", 0.0, 25.0, float(round(growth_m_default * 100, 2)), step=0.1) / 100.0
churn_input = c3.slider("Churn mensual (%)", 0.0, 15.0, float(round(churn_m_default * 100, 2)), step=0.1) / 100.0

c4, c5 = st.columns(2)
sector = c4.selectbox("Sector (múltiplo por defecto)", options=list(DEFAULT_MULTIPLES.keys()))
multiple = c5.slider("Múltiplo aplicado (x ARR)", 1.0, 20.0, float(DEFAULT_MULTIPLES[sector]), step=0.5)

years = st.slider("Horizonte de proyección (años)", 1, 5, 3)

net_monthly = (growth_input - churn_input)
net_annual = (1.0 + net_monthly) ** 12 - 1.0

m1, m2, m3 = st.columns(3)
m1.metric("ARR actual", fmt_eur(arr_input))
m2.metric("Crecimiento neto anual", f"{net_annual*100:,.1f}%".replace(",", "."))
m3.metric("Múltiplo", f"{multiple:.1f}x")

# ---------------------------
# Proyección y valoración
# ---------------------------
months = years * 12
start_idx = last_date if last_date is not None else pd.Timestamp.today().normalize()

proj_df = build_projection_df(arr_input, growth_input, churn_input, months, start_idx)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### Evolución proyectada de ARR (mensual)")
    st.line_chart(proj_df, use_container_width=True)

with col_right:
    st.markdown("### Valoración estimada")
    val3 = valuation_at_years(arr_input, growth_input, churn_input, years=3, multiple=multiple)
    val5 = valuation_at_years(arr_input, growth_input, churn_input, years=5, multiple=multiple)
    st.write("A **3 años**:", fmt_eur(val3))
    st.write("A **5 años**:", fmt_eur(val5))

    st.markdown("#### Escenarios (a 3 años)")
    base_3 = valuation_at_years(arr_input, growth_input, churn_input, 3, multiple)
    opt_3 = valuation_at_years(arr_input, growth_input*1.20, churn_input*0.75, 3, multiple)
    pes_3 = valuation_at_years(arr_input, max(growth_input*0.80, 0.0), churn_input*1.25, 3, multiple)

    scen = pd.DataFrame({
        "Escenario": ["Base", "Optimista", "Pesimista"],
        "Valoración (3 años)": [base_3, opt_3, pes_3]
    })
    scen["Valoración (3 años)"] = scen["Valoración (3 años)"].map(fmt_eur)
    st.dataframe(scen, hide_index=True, use_container_width=True)

# ---------------------------
# Exportación
# ---------------------------
st.markdown("### Exportar proyección")
export_df = proj_df.copy()
export_df.index.name = "Fecha"
export_df["Múltiplo aplicado"] = multiple
export_df["Crecimiento mensual"] = growth_input
export_df["Churn mensual"] = churn_input

csv = export_df.to_csv(index=True).encode("utf-8")
st.download_button("⬇️ Descargar CSV de la proyección", data=csv, file_name="proyeccion_saas.csv", mime="text/csv")

# Excel
buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    export_df.to_excel(writer, sheet_name="Proyección")
st.download_button("⬇️ Descargar Excel de la proyección", data=buf.getvalue(), file_name="proyeccion_saas.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Consejo: ajusta crecimiento y churn para simular entradas/salidas de clientes mensualmente. Si aportas MRR/ARR históricos en el Excel, el ARR actual se precarga con el último mes disponible.")
