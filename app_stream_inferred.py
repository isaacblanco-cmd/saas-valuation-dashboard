
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="SaaS Valuation Dashboard • Inferred Expansions", page_icon="📈", layout="wide")

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
def read_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file, sheet_name=None, engine="openpyxl")

def build_summary_from_data(df_data: pd.DataFrame) -> pd.DataFrame:
    # When only Data sheet provided, rebuild Monthly Summary
    df = df_data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    # Aggregate per month
    grp = df.groupby("Date", as_index=False).agg({
        "New Customers":"sum",
        "Lost Customers":"sum",
        "Active Customers":"sum",
        "New MRR (€)":"sum",
        "Expansion MRR (inferred €)":"sum",
        "Churned MRR (€)":"sum",
        "Downgraded MRR (inferred €)":"sum",
        "Real MRR used (€)":"sum"
    }).sort_values("Date")
    grp["Net New MRR (€)"] = grp["New MRR (€)"] + grp["Expansion MRR (inferred €)"] - grp["Churned MRR (€)"] - grp["Downgraded MRR (inferred €)"]
    grp = grp.rename(columns={"Real MRR used (€)":"Total MRR (€)"})
    grp["Total ARR (€)"] = grp["Total MRR (€)"] * 12.0
    grp["MoM Growth %"] = grp["Total MRR (€)"].pct_change().fillna(0.0)
    grp["Churn % (customers)"] = grp["Lost Customers"].div(grp["Active Customers"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    grp["ARPU (€)"] = (grp["Total MRR (€)"].div(grp["Active Customers"].replace(0, np.nan))).fillna(0.0)
    return grp

def project_monthly_arr(arr0: float, growth_m: float, churn_m: float, months: int):
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
    df = pd.DataFrame({"Base": base, "Optimista": optimistic, "Pesimista": pessimistic})
    if start_date is not None:
        idx = pd.date_range(start=start_date + pd.offsets.MonthBegin(1), periods=months, freq="MS")
        df.index = idx
    return df

def valuation_at_years(arr0: float, growth_m: float, churn_m: float, years: int, multiple: float) -> float:
    months = years * 12
    arr_n = arr0 * ((1.0 + (growth_m - churn_m)) ** months)
    return arr_n * multiple

st.title("📈 SaaS KPIs & Valuation (Inferred Expansions)")

with st.sidebar:
    st.header("Carga tu Excel / CSV")
    uploaded = st.file_uploader("Plantilla recomendada: SaaS_Simple_Inferred_Expansions_CLEAN.xlsx", type=["xlsx","xls","csv"])
    st.caption("Si subes el Excel con hojas **Data** y **Monthly_Summary**, las usaremos. Si solo hay **Data**, reconstruimos el resumen.")

df_data = None
df_summary = None
last_date = None
arr_current = None

if uploaded is not None:
    book = read_file(uploaded)
    if isinstance(book, dict):
        # Excel con hojas
        df_data = book.get("Data")
        df_summary = book.get("Monthly_Summary")
        prices = book.get("Prices")
    else:
        # CSV simple
        df_data = book
        df_summary = None
        prices = None

    if df_data is not None and not df_data.empty:
        # Normalize
        df_data["Date"] = pd.to_datetime(df_data["Date"])
        df_data = df_data.sort_values(["Date","Plan"])
        last_date = pd.to_datetime(df_data["Date"].max())
        # Rebuild summary if missing
        if df_summary is None or df_summary.empty:
            df_summary = build_summary_from_data(df_data)

        # KPIs headline
        last_row = df_summary.loc[df_summary["Date"] == df_summary["Date"].max()].iloc[0]
        arr_current = float(last_row["Total ARR (€)"])

        st.subheader("Resumen (último mes)")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MRR total", fmt_eur(last_row["Total MRR (€)"]))
        k2.metric("ARR total", fmt_eur(last_row["Total ARR (€)"]))
        k3.metric("Net New MRR", fmt_eur(last_row["Net New MRR (€)"]))
        k4.metric("ARPU", fmt_eur(last_row["ARPU (€)"]))

        with st.expander("Ver tablas crudas"):
            st.write("**Data**")
            st.dataframe(df_data.head(50), use_container_width=True)
            st.write("**Monthly_Summary**")
            st.dataframe(df_summary.tail(24), use_container_width=True)

        # Charts
        st.markdown("### Evolución (agregado)")
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(df_summary.set_index("Date")[["Total MRR (€)"]], use_container_width=True)
        with c2:
            st.area_chart(df_summary.set_index("Date")[["New MRR (€)","Expansion MRR (inferred €)","Churned MRR (€)","Downgraded MRR (inferred €)"]], use_container_width=True)

        # Per-plan breakdown for latest month
        st.markdown("### Desglose por plan (último mes)")
        latest = df_data[df_data["Date"] == last_date].copy()
        per_plan = latest.groupby("Plan", as_index=False).agg({
            "Active Customers":"sum",
            "Real MRR used (€)":"sum",
            "New MRR (€)":"sum",
            "Churned MRR (€)":"sum",
            "Expansion MRR (inferred €)":"sum",
            "Downgraded MRR (inferred €)":"sum"
        }).rename(columns={"Real MRR used (€)":"MRR"})
        per_plan["ARR"] = per_plan["MRR"] * 12.0
        per_plan["Mix %"] = per_plan["MRR"] / per_plan["MRR"].sum() * 100.0
        st.dataframe(per_plan.assign(
            MRR_fmt=per_plan["MRR"].map(fmt_eur),
            ARR_fmt=per_plan["ARR"].map(fmt_eur),
            Mix_fmt=per_plan["Mix %"].map(lambda x: f"{x:.1f}%")
        )[["Plan","Active Customers","MRR_fmt","ARR_fmt","Mix_fmt","New MRR (€)","Expansion MRR (inferred €)","Churned MRR (€)","Downgraded MRR (inferred €)"]], hide_index=True, use_container_width=True)

# ===== Valuation & Scenarios =====
st.markdown("---")
st.markdown("## Valoración y escenarios")
if arr_current is None:
    arr_current = st.number_input("ARR actual (€)", value=1_000_000.0, step=50_000.0, format="%.0f")
else:
    st.caption(f"ARR detectado desde el Excel (último mes): {fmt_eur(arr_current)}")
    arr_current = st.number_input("ARR actual (€) (editable)", value=float(arr_current), step=50_000.0, format="%.0f")

c1, c2, c3 = st.columns(3)
growth = c1.slider("Crecimiento mensual (%)", 0.0, 25.0, 5.0, step=0.1) / 100.0
churn = c2.slider("Churn mensual (%)", 0.0, 15.0, 2.0, step=0.1) / 100.0
sector = c3.selectbox("Sector (múltiplo por defecto)", options=list(DEFAULT_MULTIPLES.keys()))
multiple = st.slider("Múltiplo aplicado (x ARR)", 1.0, 20.0, float(DEFAULT_MULTIPLES[sector]), step=0.5)
years = st.slider("Horizonte de proyección (años)", 1, 5, 3)

net_annual = (1.0 + (growth - churn)) ** 12 - 1.0
m1, m2, m3 = st.columns(3)
m1.metric("Crecimiento neto anual", f"{net_annual*100:,.1f}%".replace(",", "."))
m2.metric("Múltiplo", f"{multiple:.1f}x")
m3.metric("Horizonte", f"{years} años")

months = years * 12
start_idx = (last_date if last_date is not None else pd.Timestamp.today().normalize())
proj = build_projection_df(arr_current, growth, churn, months, start_idx)

colL, colR = st.columns([2,1])
with colL:
    st.markdown("### Proyección ARR")
    st.line_chart(proj, use_container_width=True)
with colR:
    st.markdown("### Valoración estimada")
    val3 = valuation_at_years(arr_current, growth, churn, 3, multiple)
    val5 = valuation_at_years(arr_current, growth, churn, 5, multiple)
    st.write("A **3 años**:", fmt_eur(val3))
    st.write("A **5 años**:", fmt_eur(val5))

# Export projections
st.markdown("### Exportar proyección")
export_df = proj.copy()
export_df.index.name = "Fecha"
export_df["Múltiplo aplicado"] = multiple
export_df["Crecimiento mensual"] = growth
export_df["Churn mensual"] = churn

csv = export_df.to_csv(index=True).encode("utf-8")
st.download_button("⬇️ Descargar CSV", data=csv, file_name="projection.csv", mime="text/csv")

buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    export_df.to_excel(writer, sheet_name="Projection")
st.download_button("⬇️ Descargar Excel", data=buf.getvalue(), file_name="projection.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Consejo: para que el Excel alimente bien los KPIs, rellena solo New/Lost/Active y, si hay descuentos, el Real MRR del mes. Las expansiones y downgrades se infieren automáticamente.")
