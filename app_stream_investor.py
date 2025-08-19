
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Investor SaaS Dashboard", page_icon="💼", layout="wide")

DEFAULT_MULTIPLES = {
    "Academy/Starter": 6.0,
    "Basic": 7.0,
    "Advance": 8.0,
    "Pro": 9.0,
    "SaaS Infra / DevTools": 10.0,
    "FinTech": 12.0,
    "HealthTech": 8.0,
    "Horizontal B2B SaaS": 8.0,
    "Vertical SaaS": 6.0,
    "EdTech": 6.0,
    "E-commerce Enablement": 5.0,
    "Otro": 7.0,
}

def fmt_eur(x):
    try:
        return f"{x:,.0f} €".replace(",", ".")
    except Exception:
        return str(x)

@st.cache_data
def read_book(file):
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
        return {"Data": df}
    else:
        return pd.read_excel(file, sheet_name=None, engine="openpyxl")

def rebuild_summary(df_data: pd.DataFrame) -> pd.DataFrame:
    df = df_data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    # Fallbacks if columns missing
    for col in ["Active Used","New MRR (€)","Expansion MRR (inferred €)","Churned MRR (€)","Downgraded MRR (inferred €)","Real MRR used (€)"]:
        if col not in df.columns:
            df[col] = 0.0
    grp = df.groupby("Date", as_index=False).agg({
        "New Customers":"sum",
        "Lost Customers":"sum",
        "Active Used":"sum",
        "New MRR (€)":"sum",
        "Expansion MRR (inferred €)":"sum",
        "Churned MRR (€)":"sum",
        "Downgraded MRR (inferred €)":"sum",
        "Real MRR used (€)":"sum",
    }).sort_values("Date")
    grp = grp.rename(columns={
        "Active Used":"Active Customers",
        "Real MRR used (€)":"Total MRR (€)",
    })
    grp["Net New MRR (€)"] = grp["New MRR (€)"] + grp["Expansion MRR (inferred €)"] - grp["Churned MRR (€)"] - grp["Downgraded MRR (inferred €)"]
    grp["Total ARR (€)"] = grp["Total MRR (€)"] * 12.0
    grp["Start MRR (€)"] = grp["Total MRR (€)"].shift(1).fillna(0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        grp["GRR %"] = np.where(grp["Start MRR (€)"]>0, 1.0 - (grp["Churned MRR (€)"] + grp["Downgraded MRR (inferred €)"]) / grp["Start MRR (€)"], np.nan)
        grp["NRR %"] = np.where(grp["Start MRR (€)"]>0, 1.0 + (grp["Expansion MRR (inferred €)"] - (grp["Churned MRR (€)"] + grp["Downgraded MRR (inferred €)"])) / grp["Start MRR (€)"], np.nan)
        grp["MoM Growth %"] = grp["Total MRR (€)"].pct_change().fillna(0.0)
        grp["Churn % (customers)"] = grp["Lost Customers"].div(grp["Active Customers"].shift(1)).replace([np.inf,-np.inf], np.nan)
    grp["ARPU (€)"] = grp["Total MRR (€)"] / grp["Active Customers"].replace(0, np.nan)
    grp["ARPU (€)"] = grp["ARPU (€)"].fillna(0.0)
    grp["Quick Ratio"] = (grp["New MRR (€)"] + grp["Expansion MRR (inferred €)"]) / (grp["Churned MRR (€)"] + grp["Downgraded MRR (inferred €)"]).replace(0, np.nan)
    grp["Quick Ratio"] = grp["Quick Ratio"].fillna(np.inf)
    return grp

def build_cohorts_fifo(df_data: pd.DataFrame, plan: str | None):
    """Return (count_retention_df, retention_pct_df). FIFO: allocate losses to oldest cohorts first."""
    df = df_data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if plan and plan != "All plans":
        df = df[df["Plan"] == plan]
    df = df.sort_values("Date")

    # Aggregate per month (across plans if None)
    m = df.groupby("Date", as_index=False).agg({"New Customers":"sum","Lost Customers":"sum"})
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Cohort dict: key=cohort_date (first day), value=remaining count
    cohorts = []
    # We will store per month snapshot of cohort survivors by cohort year
    records = []  # rows: {"Date": date, "cohort_year": y, "age": k_months, "survivors": n}

    cohort_sizes = {}  # cohort_date -> remaining
    cohort_dates = []  # to maintain order

    unique_dates = m["Date"].tolist()
    for idx, d in enumerate(unique_dates):
        new = int(m.loc[m["Date"] == d, "New Customers"].sum())
        lost = int(m.loc[m["Date"] == d, "Lost Customers"].sum())

        # Add new cohort
        if new > 0:
            cohort_date = pd.Timestamp(d)
            cohort_sizes[cohort_date] = cohort_sizes.get(cohort_date, 0) + new
            cohort_dates.append(cohort_date)

        # Apply losses FIFO
        to_remove = lost
        i = 0
        while to_remove > 0 and i < len(cohort_dates):
            cdate = cohort_dates[i]
            remain = cohort_sizes.get(cdate, 0)
            if remain <= 0:
                i += 1
                continue
            take = min(remain, to_remove)
            cohort_sizes[cdate] = remain - take
            to_remove -= take
            if cohort_sizes[cdate] == 0:
                i += 1
        # prune zero cohorts at front
        cohort_dates = [cd for cd in cohort_dates if cohort_sizes.get(cd, 0) > 0 or cd in cohort_sizes]

        # snapshot survivors by cohort year and age (in months)
        for cdate, rem in cohort_sizes.items():
            if rem <= 0:
                continue
            age = (d.year - cdate.year) * 12 + (d.month - cdate.month)
            records.append({
                "Date": d,
                "cohort": pd.Timestamp(cdate),
                "cohort_year": cdate.year,
                "age": age,
                "survivors": rem
            })

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    snap = pd.DataFrame(records)
    # Build retention table by cohort_year and age (sum over cohorts in that year)
    cohort_year_age = snap.groupby(["cohort_year","age"], as_index=False)["survivors"].sum()
    # initial sizes by cohort_year
    init_sizes = m.copy()
    init_sizes["cohort_year"] = init_sizes["Date"].dt.year
    init_sizes = init_sizes.groupby("cohort_year", as_index=False)["New Customers"].sum().rename(columns={"New Customers":"init"})
    # pivot survivors
    mat = cohort_year_age.pivot_table(index="cohort_year", columns="age", values="survivors", aggfunc="sum").fillna(0).sort_index()
    # include age 0 initial sizes in matrix
    if 0 not in mat.columns:
        mat[0] = 0
    mat = mat.sort_index(axis=1)
    # Set column 0 to initial size (cohort formation)
    mat[0] = init_sizes.set_index("cohort_year")["init"].reindex(mat.index).fillna(0).values
    # Build retention pct
    denom = mat[[0]].replace(0, np.nan)
    ret_pct = mat.divide(denom.values, axis=0).fillna(0.0)
    return mat.astype(int), ret_pct

def ytd_filter(df: pd.DataFrame, years: list[int], ytd: bool):
    if df is None or df.empty:
        return df
    out = df[df["Date"].dt.year.isin(years)].copy()
    if ytd:
        today = pd.Timestamp.today().normalize().replace(day=1)  # up to current month
        out = out[out["Date"] <= today]
    return out

st.title("💼 SaaS Investor Dashboard (YTD, Cohorts, NRR/GRR, Multiples per Plan)")

with st.sidebar:
    st.header("Upload file")
    up = st.file_uploader("Excel/CSV. Recommended: SaaS_Final_Template.xlsx", type=["xlsx","xls","csv"])
    st.caption("If Monthly_Summary is missing, it will be rebuilt from Data.")

if not up:
    st.info("Upload your spreadsheet to begin.")
    st.stop()

book = read_book(up)
df_data = book.get("Data")
df_sum = book.get("Monthly_Summary")
df_prices = book.get("Prices")

if df_data is None or df_data.empty:
    st.error("No 'Data' sheet found or it is empty.")
    st.stop()

# Normalize
df_data["Date"] = pd.to_datetime(df_data["Date"])
df_data = df_data.sort_values(["Date","Plan"])

# Rebuild summary if needed
if df_sum is None or df_sum.empty or "Total MRR (€)" not in df_sum.columns:
    df_sum = rebuild_summary(df_data)
else:
    df_sum["Date"] = pd.to_datetime(df_sum["Date"])
    df_sum = df_sum.sort_values("Date")
    # add NRR/GRR if missing
    if "Start MRR (€)" not in df_sum.columns:
        df_sum["Start MRR (€)"] = df_sum["Total MRR (€)"].shift(1).fillna(0.0)
    if "GRR %" not in df_sum.columns or "NRR %" not in df_sum.columns:
        df_sum["GRR %"] = np.where(df_sum["Start MRR (€)"]>0, 1.0 - (df_sum["Churned MRR (€)"] + df_sum["Downgraded MRR (inferred €)"]) / df_sum["Start MRR (€)"], np.nan)
        df_sum["NRR %"] = np.where(df_sum["Start MRR (€)"]>0, 1.0 + (df_sum["Expansion MRR (inferred €)"] - (df_sum["Churned MRR (€)"] + df_sum["Downgraded MRR (inferred €)"])) / df_sum["Start MRR (€)"], np.nan)
    if "ARPU (€)" not in df_sum.columns and "Active Customers" in df_sum.columns:
        df_sum["ARPU (€)"] = df_sum["Total MRR (€)"] / df_sum["Active Customers"].replace(0, np.nan)

# ===== Filters =====
years_all = sorted(df_sum["Date"].dt.year.unique().tolist())
default_years = [max(years_all)]
sel_years = st.multiselect("Filter by year(s)", options=years_all, default=default_years, help="Choose one or multiple years to analyze.")
ytd = st.toggle("Año actual (YTD)", value=True, help="If ON, restricts to months up to current month in the latest selected year(s).")
df_sum_f = ytd_filter(df_sum, sel_years, ytd)

# ===== KPIs Headline =====
if df_sum_f.empty:
    st.warning("No data in the selected filter range.")
    st.stop()

last_row = df_sum_f.iloc[-1]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Active customers", f"{int(last_row.get('Active Customers', 0)):,}".replace(",", "."))
k2.metric("MRR total", fmt_eur(last_row["Total MRR (€)"]))
k3.metric("ARR total", fmt_eur(last_row["Total ARR (€)"]))
k4.metric("Net New MRR (last)", fmt_eur(last_row["Net New MRR (€)"]))
# YTD growth
first_row_year = df_sum_f.iloc[0]
ytd_growth = (last_row["Total MRR (€)"] - first_row_year["Total MRR (€)"]) / first_row_year["Total MRR (€)"] if first_row_year["Total MRR (€)"]>0 else np.nan
k5.metric("Growth YTD", f"{(ytd_growth*100):.1f}%" if pd.notna(ytd_growth) else "—")

# Quick Ratio YTD
sum_new = df_sum_f["New MRR (€)"].sum()
sum_exp = df_sum_f["Expansion MRR (inferred €)"].sum()
sum_churn = df_sum_f["Churned MRR (€)"].sum()
sum_down = df_sum_f["Downgraded MRR (inferred €)"].sum()
qr_ytd = (sum_new + sum_exp) / (sum_churn + sum_down) if (sum_churn + sum_down) > 0 else np.inf

c1, c2, c3 = st.columns(3)
c1.metric("Quick Ratio (YTD)", f"{qr_ytd:.2f}" if np.isfinite(qr_ytd) else "∞")
grr_ytd = np.prod(df_sum_f["GRR %"].dropna().values) if (~df_sum_f["GRR %"].isna()).any() else np.nan
nrr_ytd = np.prod(df_sum_f["NRR %"].dropna().values) if (~df_sum_f["NRR %"].isna()).any() else np.nan
c2.metric("GRR YTD", f"{(grr_ytd-1)*100:.1f}%" if pd.notna(grr_ytd) else "—")
c3.metric("NRR YTD", f"{(nrr_ytd-1)*100:.1f}%" if pd.notna(nrr_ytd) else "—")

st.markdown("### Evolution")
left, right = st.columns(2)
with left:
    st.line_chart(df_sum_f.set_index("Date")[["Total MRR (€)","Total ARR (€)"]], use_container_width=True)
with right:
    st.area_chart(df_sum_f.set_index("Date")[["New MRR (€)","Expansion MRR (inferred €)","Churned MRR (€)","Downgraded MRR (inferred €)"]], use_container_width=True)

# ===== Per plan breakdown for latest month in filter =====
st.markdown("### Per plan (latest in filter)")
latest_date = df_sum_f["Date"].max()
latest_plans = df_data[df_data["Date"] == latest_date].copy()
if "Real MRR used (€)" not in latest_plans.columns:
    latest_plans["Real MRR used (€)"] = 0.0
if "Active Used" not in latest_plans.columns and "Active Customers" in latest_plans.columns:
    latest_plans["Active Used"] = latest_plans["Active Customers"]

plan_tbl = latest_plans.groupby("Plan", as_index=False).agg({
    "Active Used":"sum",
    "Real MRR used (€)":"sum",
    "New MRR (€)":"sum",
    "Churned MRR (€)":"sum",
    "Expansion MRR (inferred €)":"sum",
    "Downgraded MRR (inferred €)":"sum",
})
plan_tbl["ARR (€)"] = plan_tbl["Real MRR used (€)"] * 12.0
plan_tbl["Mix %"] = plan_tbl["Real MRR used (€)"] / plan_tbl["Real MRR used (€)"].sum() * 100.0
st.dataframe(plan_tbl.assign(
    MRR_fmt=plan_tbl["Real MRR used (€)"].map(fmt_eur),
    ARR_fmt=plan_tbl["ARR (€)"].map(fmt_eur),
    Mix_fmt=plan_tbl["Mix %"].map(lambda v: f"{v:.1f}%"),
)[["Plan","Active Used","MRR_fmt","ARR_fmt","Mix_fmt","New MRR (€)","Expansion MRR (inferred €)","Churned MRR (€)","Downgraded MRR (inferred €)"]], hide_index=True, use_container_width=True)

# ===== Valuation: multiples per plan =====
st.markdown("---")
st.markdown("## Valuation (multiples per plan)")

if df_prices is None or df_prices.empty:
    df_mult = pd.DataFrame({"Plan": plan_tbl["Plan"], "Multiple (x ARR)": [DEFAULT_MULTIPLES.get(p, 7.0) for p in plan_tbl["Plan"]]})
else:
    df_mult = df_prices[["Plan","Multiple (x ARR)"]].copy() if "Multiple (x ARR)" in df_prices.columns else pd.DataFrame({"Plan": df_prices["Plan"], "Multiple (x ARR)": [DEFAULT_MULTIPLES.get(p, 7.0) for p in df_prices["Plan"]]})
    # Ensure all plans in table
    for p in plan_tbl["Plan"]:
        if p not in df_mult["Plan"].values:
            df_mult.loc[len(df_mult)] = {"Plan": p, "Multiple (x ARR)": DEFAULT_MULTIPLES.get(p, 7.0)}

edited = st.data_editor(df_mult, num_rows="dynamic", use_container_width=True, key="multiples_editor",
                        help="Adjust multiples per plan. These are applied to ARR (latest month of the filter).")

val_tbl = plan_tbl[["Plan","ARR (€)"]].merge(edited, on="Plan", how="left")
val_tbl["Multiple (x ARR)"] = val_tbl["Multiple (x ARR)"].fillna(val_tbl["Plan"].map(lambda p: DEFAULT_MULTIPLES.get(p, 7.0)))
val_tbl["Valuation (€)"] = val_tbl["ARR (€)"] * val_tbl["Multiple (x ARR)"]
st.dataframe(val_tbl.assign(
    ARR_fmt=val_tbl["ARR (€)"].map(fmt_eur),
    Multiple_fmt=val_tbl["Multiple (x ARR)"].map(lambda x: f"{x:.1f}x"),
    Valu_fmt=val_tbl["Valuation (€)"].map(fmt_eur),
)[["Plan","ARR_fmt","Multiple_fmt","Valu_fmt"]], hide_index=True, use_container_width=True)

st.metric("Valuation (Total)", fmt_eur(val_tbl["Valuation (€)"].sum()))

# ===== Cohorts =====
st.markdown("---")
st.markdown("## Cohorts (counts, FIFO by month)")
plan_opt = ["All plans"] + sorted(df_data["Plan"].dropna().unique().tolist())
sel_plan = st.selectbox("Plan for cohort view", options=plan_opt, index=0, help="Cohort analysis uses New/Lost counts and FIFO allocation of churn.")

mat_counts, mat_pct = build_cohorts_fifo(df_data, plan=None if sel_plan == "All plans" else sel_plan)
if mat_counts.empty:
    st.info("Not enough data to build cohorts.")
else:
    st.caption("Rows = Cohort Year, Columns = Age in months. Values = survivors (counts) / retention %.")
    st.dataframe(mat_counts, use_container_width=True)
    st.dataframe((mat_pct*100).round(1).astype(float), use_container_width=True)

# ===== Export snapshot for investors =====
st.markdown("---")
st.markdown("### Export snapshot (filtered)")
snap = df_sum_f.copy()
buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as w:
    snap.to_excel(w, sheet_name="Monthly_Summary_Filtered", index=False)
    plan_tbl.to_excel(w, sheet_name="Per_Plan_Latest", index=False)
    val_tbl.to_excel(w, sheet_name="Valuation_By_Plan", index=False)
st.download_button("⬇️ Download snapshot (Excel)", data=buf.getvalue(),
                   file_name="investor_snapshot.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Tip: Use the year filter and YTD toggle to prepare investor updates, then export the snapshot.")
